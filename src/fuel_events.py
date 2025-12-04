"""
Fuel Events Processor
Algorithm 2: Smoothed fuel data → Event Detection → Clustering → Validation

Uses configuration from: config/fuel_events.etl.json

Input: Processed fuel data from FuelSensorDataProcessor
Output: silver_data_layer.fuel_events
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import json

from .config import FuelEventsETLConfig, load_fuel_events_config
from .database import ObjectInfo
from .detection import (
    detect_fuel_events,
    events_to_dataframe,
    FuelEventCluster,
    CandidatePoint
)


class FuelEventsProcessor:
    """
    Processor for fuel events detection (Algorithm 2).
    
    Pipeline:
    1. Take processed fuel sensor data (smoothed, in liters)
    2. Detect candidate points (significant fuel level changes)
    3. Cluster nearby candidates into events
    4. Validate events with context (speed, ignition, location)
    5. Filter by minimum volume threshold
    
    Config: fuel_events.etl.json
    Output: silver_data_layer.fuel_events
    """
    
    def __init__(
        self,
        config: Optional[FuelEventsETLConfig] = None
    ):
        """
        Initialize processor.
        
        Args:
            config: ETL configuration (loads default if not provided)
        """
        self.config = config or load_fuel_events_config()
        
        # Processing state
        self.fuel_data: Optional[pd.DataFrame] = None
        self.events: List[FuelEventCluster] = []
        self.candidate_points: List[CandidatePoint] = []
        self.object_info: Optional[ObjectInfo] = None
    
    def load_fuel_data(
        self,
        fuel_data: pd.DataFrame,
        object_info: ObjectInfo
    ) -> None:
        """
        Load processed fuel sensor data.
        
        Args:
            fuel_data: Processed fuel sensor DataFrame (from FuelSensorDataProcessor)
            object_info: Object information
        """
        self.fuel_data = fuel_data.copy()
        self.object_info = object_info
    
    def detect_events(self) -> List[FuelEventCluster]:
        """
        Detect fuel events from processed data.
        
        Uses plateau-based volume calculation:
        1. Detects candidate points (significant fuel changes)
        2. Clusters nearby candidates
        3. Extends boundaries to find stable plateau levels
        4. Calculates volume as difference between plateaus
        
        Returns:
            List of detected events (refuels and drains)
        """
        if self.fuel_data is None or self.fuel_data.empty:
            raise ValueError("No fuel data loaded. Call load_fuel_data() first.")
        
        tank_volume = None
        if self.object_info and self.object_info.fuel_tank_volume:
            tank_volume = self.object_info.fuel_tank_volume
        
        # Build config dict for detection (includes plateau parameters)
        config_dict = {
            "detection": self.config.parameters.detection.model_dump(),
            "context": self.config.parameters.context.model_dump()
        }
        
        self.events = detect_fuel_events(
            df=self.fuel_data,
            config=config_dict,
            value_column="fuel_level_l",
            time_column="device_time",
            tank_volume=tank_volume
        )
        
        return self.events
    
    def get_silver_layer_dataframe(self) -> pd.DataFrame:
        """
        Get events in silver_data_layer.fuel_events format.
        
        Returns:
            DataFrame matching silver layer schema
        """
        if not self.events:
            return pd.DataFrame()
        
        device_id = self.fuel_data["device_id"].iloc[0] if self.fuel_data is not None else None
        object_id = self.object_info.object_id if self.object_info else None
        vehicle_id = self.object_info.vehicle_id if self.object_info else None
        
        return events_to_dataframe(
            events=self.events,
            device_id=device_id,
            object_id=object_id,
            vehicle_id=vehicle_id,
            processing_version=self.config.get_processing_version(),
            processing_params=self._get_processing_params()
        )
    
    def _get_processing_params(self) -> dict:
        """Get processing parameters for storage"""
        return {
            "config_version": self.config.version,
            "model_name": self.config.metadata.model_name,
            "model_version": self.config.metadata.model_version,
            "detection_mode": self.config.parameters.detection.mode,
            "min_volume_l": self.config.parameters.detection.event.min_volume_l
        }
    
    def save_to_csv(self, output_path: str | Path) -> str:
        """
        Save events to CSV file.
        
        Args:
            output_path: Path for output file
        
        Returns:
            Path to saved file
        """
        events_df = self.get_silver_layer_dataframe()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        events_df.to_csv(output_path, index=False)
        
        return str(output_path)
    
    def get_summary(self) -> dict:
        """Get summary of detected events"""
        if not self.events:
            summary = {
                "total_events": 0,
                "refuels": 0,
                "drains": 0,
                "total_refuel_volume": 0,
                "total_drain_volume": 0
            }
        else:
            refuels = [e for e in self.events if e.event_type == "refuel"]
            drains = [e for e in self.events if e.event_type == "drain"]
            
            summary = {
                "total_events": len(self.events),
                "refuels": len(refuels),
                "drains": len(drains),
                "total_refuel_volume": sum(e.volume_change_l for e in refuels),
                "total_drain_volume": sum(e.volume_change_l for e in drains),
                "avg_refuel_volume": float(np.mean([e.volume_change_l for e in refuels])) if refuels else 0,
                "avg_drain_volume": float(np.mean([e.volume_change_l for e in drains])) if drains else 0,
                "high_confidence_events": len([e for e in self.events if e.confidence >= 0.8])
            }
        
        # Add config info
        summary["config"] = {
            "version": self.config.version,
            "model_name": self.config.metadata.model_name,
            "target_table": f"{self.config.target.schema_name}.{self.config.target.table}"
        }
        
        return summary
