"""
Fuel Sensor Data Processor
Algorithm 1: Raw sensor data → Calibration → Zero Handling → Smoothing → Liters

Uses configuration from: config/fuel_sensor_data.etl.json

Output: silver_data_layer.fuel_sensor_data
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import json

from .config import FuelSensorDataETLConfig, load_fuel_sensor_data_config
from .database import DatabaseConnector, ObjectInfo, SensorInfo, find_data_gaps
from .calibration import convert_to_calibrated, normalize_fuel_level
from .zero_handling import handle_zeros_v2
from .smoothing import apply_smoothing_to_dataframe, SmoothingStats


class FuelSensorDataProcessor:
    """
    Processor for fuel sensor data (Algorithm 1).
    
    Pipeline:
    1. Extract raw data from Bronze layer
    2. Convert to calibrated units using calibration table
    3. Handle zero values (sticky zeros, grace period)
    4. Apply smoothing (Hampel → Median → EMA)
    5. Normalize to tank capacity
    
    Config: fuel_sensor_data.etl.json
    Output: silver_data_layer.fuel_sensor_data
    """
    
    def __init__(
        self,
        config: Optional[FuelSensorDataETLConfig] = None,
        db_connector: Optional[DatabaseConnector] = None
    ):
        """
        Initialize processor.
        
        Args:
            config: ETL configuration (loads default if not provided)
            db_connector: Database connector (optional for local testing)
        """
        self.config = config or load_fuel_sensor_data_config()
        self.db_connector = db_connector
        
        # Processing state
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.sensors: List[SensorInfo] = []
        self.object_info: Optional[ObjectInfo] = None
        self.smoothing_stats: Optional[SmoothingStats] = None
        self.data_gaps: List[Tuple[datetime, datetime]] = []
        
        # Measurement units (set during processing)
        self.measurement_units: str = "L"  # Default to liters
        self.measurement_units_description: str = "Liters"
    
    def load_data_from_db(
        self,
        object_info: ObjectInfo,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Load raw fuel data from database.
        
        Args:
            object_info: Object to load data for
            start_time: Start of time range
            end_time: End of time range
        
        Returns:
            DataFrame with raw data
        """
        if not self.db_connector:
            raise ValueError("Database connector not configured")
        
        self.object_info = object_info
        
        self.raw_data, self.sensors = self.db_connector.get_complete_fuel_data(
            object_info=object_info,
            start_time=start_time,
            end_time=end_time
        )
        
        # Find data gaps
        if not self.raw_data.empty:
            self.data_gaps = find_data_gaps(self.raw_data, "device_time", max_gap_minutes=30)
        
        return self.raw_data
    
    def load_data_from_dataframe(
        self,
        df: pd.DataFrame,
        sensors: List[SensorInfo],
        object_info: ObjectInfo
    ) -> pd.DataFrame:
        """
        Load data from existing DataFrame (for testing).
        
        Args:
            df: DataFrame with raw data
            sensors: Sensor configurations
            object_info: Object information
        
        Returns:
            Input DataFrame
        """
        self.raw_data = df.copy()
        self.sensors = sensors
        self.object_info = object_info
        
        if not self.raw_data.empty:
            self.data_gaps = find_data_gaps(self.raw_data, "device_time", max_gap_minutes=30)
        
        return self.raw_data
    
    def process(self) -> Tuple[pd.DataFrame, SmoothingStats]:
        """
        Process raw data through the full pipeline.
        
        Pipeline: Calibration → Zero Handling → Smoothing (Hampel → Median → EMA)
        
        Returns:
            Tuple of (processed DataFrame, smoothing stats)
        """
        if self.raw_data is None or self.raw_data.empty:
            raise ValueError("No data loaded. Call load_data_from_db() or load_data_from_dataframe() first.")
        
        if not self.sensors:
            raise ValueError("No fuel sensors configured")
        
        # Use first fuel sensor
        sensor = self.sensors[0]
        
        # Step 1: Convert using calibration table (values outside range are clipped)
        df, self.measurement_units, self.measurement_units_description = convert_to_calibrated(
            self.raw_data,
            sensor,
            value_column="value"
        )
        
        # Step 2: Handle zeros
        zero_config = self.config.parameters.zero_handling
        df, zero_stats = handle_zeros_v2(
            df,
            value_column="fuel_level_l_raw",
            time_column="device_time",
            mode=zero_config.mode,
            startup_grace_period_min=zero_config.startup_grace_period_min,
            min_zero_run_min=zero_config.min_zero_run_min,
            max_zero_level_l=zero_config.max_zero_level_l
        )
        
        # Step 3: Apply smoothing (Hampel → Median → EMA)
        smoothing_dict = {
            "smoothing": self.config.parameters.smoothing.model_dump()
        }
        df, self.smoothing_stats = apply_smoothing_to_dataframe(
            df,
            smoothing_dict,
            value_column="fuel_level_l_raw",
            time_column="device_time",
            output_column="fuel_level_l"
        )
        
        # Step 4: Normalize to tank volume
        tank_volume = self.object_info.fuel_tank_volume if self.object_info else None
        df["fuel_level_norm"] = df["fuel_level_l"].apply(
            lambda x: normalize_fuel_level(x, tank_volume) if pd.notna(x) else None
        )
        
        # Add metadata
        df["processing_version"] = self.config.get_processing_version()
        df["processing_params"] = json.dumps(self._get_processing_params())
        
        self.processed_data = df
        
        return df, self.smoothing_stats
    
    def _get_processing_params(self) -> dict:
        """Get processing parameters for storage"""
        return {
            "config_version": self.config.version,
            "model_name": self.config.metadata.model_name,
            "model_version": self.config.metadata.model_version,
            "smoothing_level": self.config.parameters.smoothing.smoothing_level,
            "zero_handling_mode": self.config.parameters.zero_handling.mode
        }
    
    def get_silver_layer_dataframe(self) -> pd.DataFrame:
        """
        Get processed data in silver_data_layer.fuel_sensor_data format.
        
        Returns:
            DataFrame matching silver layer schema
        """
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process() first.")
        
        df = self.processed_data.copy()
        
        # Map to silver layer columns
        silver_df = pd.DataFrame({
            "device_id": df["device_id"],
            "object_id": df.get("object_id"),
            "vehicle_id": df.get("vehicle_id"),
            "ts_utc": df["device_time"],
            "lat": df.get("lat"),
            "lng": df.get("lng"),
            "sensor_raw_value": df.get("sensor_raw_value"),  # Original raw sensor value
            "fuel_level_l_raw": df["fuel_level_l_raw"],      # After calibration
            "fuel_level_l": df["fuel_level_l"],              # After smoothing
            "fuel_level_norm": df.get("fuel_level_norm"),
            "calibration_used": df["calibration_used"],
            "calibration_points": df.get("calibration_points", 0),
            "processing_version": df["processing_version"],
            "processing_params": df["processing_params"]
        })
        
        return silver_df
    
    def save_to_csv(self, output_path: str | Path) -> str:
        """
        Save processed data to CSV file.
        
        Args:
            output_path: Path for output file
        
        Returns:
            Path to saved file
        """
        silver_df = self.get_silver_layer_dataframe()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        silver_df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def get_summary(self) -> dict:
        """Get summary of processing results"""
        if self.processed_data is None:
            return {}
        
        df = self.processed_data
        
        # Calculate clipping stats
        clipped_low_count = int(df.get("calibration_clipped_low", pd.Series([False])).sum())
        clipped_high_count = int(df.get("calibration_clipped_high", pd.Series([False])).sum())
        total_valid = int(df["fuel_level_l"].notna().sum())
        
        return {
            "total_points": len(df),
            "valid_points": total_valid,
            "calibration_used": bool(df["calibration_used"].any()),
            "calibration_points": int(df["calibration_points"].max()) if "calibration_points" in df else 0,
            "measurement_units": self.measurement_units,
            "measurement_units_description": self.measurement_units_description,
            "min_fuel": float(df["fuel_level_l"].min()) if df["fuel_level_l"].notna().any() else 0,
            "max_fuel": float(df["fuel_level_l"].max()) if df["fuel_level_l"].notna().any() else 0,
            "avg_fuel": float(df["fuel_level_l"].mean()) if df["fuel_level_l"].notna().any() else 0,
            "calibration_clipped_low": clipped_low_count,
            "calibration_clipped_high": clipped_high_count,
            "calibration_clipped_total": clipped_low_count + clipped_high_count,
            "calibration_clipped_pct": round((clipped_low_count + clipped_high_count) / total_valid * 100, 1) if total_valid > 0 else 0,
            "data_gaps_count": len(self.data_gaps),
            "smoothing_stats": {
                "outliers_removed": self.smoothing_stats.outliers_removed if self.smoothing_stats else 0,
                "median_window": self.smoothing_stats.median_window_used if self.smoothing_stats else 0,
                "ema_alpha": self.smoothing_stats.ema_alpha_used if self.smoothing_stats else 0
            },
            "config": {
                "version": self.config.version,
                "model_name": self.config.metadata.model_name,
                "smoothing_level": self.config.parameters.smoothing.smoothing_level,
                "target_table": f"{self.config.target.schema_name}.{self.config.target.table}"
            }
        }
