"""
Calibration and unit conversion module
Converts raw sensor values to liters using calibration tables
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .database import SensorInfo


@dataclass
class CalibrationResult:
    """Result of calibration conversion"""
    value_liters: float
    calibration_used: bool
    calibration_points: int
    interpolation_type: str  # "exact", "interpolated", "extrapolated", "fallback"


class CalibrationTable:
    """
    Calibration table for converting raw sensor values to liters.
    Uses linear interpolation/extrapolation.
    """
    
    def __init__(self, calibration_data: Optional[List[Dict[str, float]]] = None):
        """
        Initialize calibration table.
        
        Args:
            calibration_data: List of {"in": raw_value, "out": liters} dictionaries
        """
        self.points: List[Tuple[float, float]] = []
        self._in_values: np.ndarray = np.array([])
        self._out_values: np.ndarray = np.array([])
        
        if calibration_data:
            self._load_calibration_data(calibration_data)
    
    def _load_calibration_data(self, data: List[Dict[str, float]]) -> None:
        """Load and sort calibration data"""
        if not data:
            return
        
        # Extract and sort by input value
        points = [(float(d["in"]), float(d["out"])) for d in data if "in" in d and "out" in d]
        points.sort(key=lambda x: x[0])
        
        self.points = points
        self._in_values = np.array([p[0] for p in points])
        self._out_values = np.array([p[1] for p in points])
    
    @property
    def is_valid(self) -> bool:
        """Check if calibration table has enough points"""
        return len(self.points) >= 2
    
    @property
    def num_points(self) -> int:
        """Number of calibration points"""
        return len(self.points)
    
    @property
    def range(self) -> Tuple[float, float]:
        """Get valid input range (min, max)"""
        if not self.is_valid:
            return (0.0, 0.0)
        return (self._in_values[0], self._in_values[-1])
    
    def convert(self, raw_value: float) -> CalibrationResult:
        """
        Convert raw sensor value to liters using calibration table.
        
        Args:
            raw_value: Raw sensor reading
        
        Returns:
            CalibrationResult with converted value and metadata
        """
        if not self.is_valid:
            return CalibrationResult(
                value_liters=raw_value,
                calibration_used=False,
                calibration_points=0,
                interpolation_type="fallback"
            )
        
        # Check for exact match
        exact_idx = np.where(np.isclose(self._in_values, raw_value, rtol=1e-6))[0]
        if len(exact_idx) > 0:
            return CalibrationResult(
                value_liters=float(self._out_values[exact_idx[0]]),
                calibration_used=True,
                calibration_points=self.num_points,
                interpolation_type="exact"
            )
        
        # Determine interpolation type
        min_in, max_in = self.range
        if raw_value < min_in or raw_value > max_in:
            interp_type = "extrapolated"
        else:
            interp_type = "interpolated"
        
        # Linear interpolation/extrapolation
        result = np.interp(raw_value, self._in_values, self._out_values)
        
        # For extrapolation beyond range, use linear extension
        if raw_value < min_in and len(self._in_values) >= 2:
            # Extrapolate using first two points
            slope = (self._out_values[1] - self._out_values[0]) / (self._in_values[1] - self._in_values[0])
            result = self._out_values[0] + slope * (raw_value - self._in_values[0])
        elif raw_value > max_in and len(self._in_values) >= 2:
            # Extrapolate using last two points
            slope = (self._out_values[-1] - self._out_values[-2]) / (self._in_values[-1] - self._in_values[-2])
            result = self._out_values[-1] + slope * (raw_value - self._in_values[-1])
        
        return CalibrationResult(
            value_liters=float(result),
            calibration_used=True,
            calibration_points=self.num_points,
            interpolation_type=interp_type
        )
    
    def convert_array(self, raw_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert array of raw values to liters.
        
        Returns:
            Tuple of (liters_array, is_extrapolated_mask)
        """
        if not self.is_valid:
            return raw_values.copy(), np.ones(len(raw_values), dtype=bool)
        
        liters = np.interp(raw_values, self._in_values, self._out_values)
        
        # Handle extrapolation
        min_in, max_in = self.range
        below_mask = raw_values < min_in
        above_mask = raw_values > max_in
        
        if np.any(below_mask) and len(self._in_values) >= 2:
            slope = (self._out_values[1] - self._out_values[0]) / (self._in_values[1] - self._in_values[0])
            liters[below_mask] = self._out_values[0] + slope * (raw_values[below_mask] - self._in_values[0])
        
        if np.any(above_mask) and len(self._in_values) >= 2:
            slope = (self._out_values[-1] - self._out_values[-2]) / (self._in_values[-1] - self._in_values[-2])
            liters[above_mask] = self._out_values[-1] + slope * (raw_values[above_mask] - self._in_values[-1])
        
        extrapolated = below_mask | above_mask
        
        return liters, extrapolated


def apply_sensor_parameters(
    raw_value: float,
    sensor_info: SensorInfo
) -> Optional[float]:
    """
    Apply sensor parameters to raw value.
    
    Applies:
        - less/more thresholds (from parameters)
        - multiplier/divider
    
    Returns:
        Processed value or None if value is outside thresholds
    """
    # Check thresholds from parameters
    params = sensor_info.parameters or {}
    
    less_threshold = params.get("less")
    more_threshold = params.get("more")
    
    # Filter out values outside thresholds
    if less_threshold is not None and raw_value < float(less_threshold):
        return None
    if more_threshold is not None and raw_value > float(more_threshold):
        return None
    
    # Apply multiplier and divider
    result = raw_value
    
    if sensor_info.multiplier is not None and sensor_info.multiplier != 0:
        result *= sensor_info.multiplier
    
    if sensor_info.divider is not None and sensor_info.divider != 0:
        result /= sensor_info.divider
    
    return result


def convert_to_liters(
    df: pd.DataFrame,
    sensor_info: SensorInfo,
    value_column: str = "value"
) -> pd.DataFrame:
    """
    Convert raw sensor data to liters using calibration table.
    
    Args:
        df: DataFrame with raw sensor values
        sensor_info: Sensor configuration with calibration data
        value_column: Name of column with raw values
    
    Returns:
        DataFrame with additional columns:
            - sensor_raw_value: Original raw sensor value (before calibration)
            - fuel_level_l_raw: Calibrated value in liters
            - calibration_used: Whether calibration was applied
            - calibration_points: Number of calibration points used
    """
    result_df = df.copy()
    
    # Parse raw values to numeric
    result_df["raw_numeric"] = pd.to_numeric(result_df[value_column], errors="coerce")
    
    # Apply sensor parameters (thresholds, multiplier, divider)
    def apply_params(row):
        if pd.isna(row["raw_numeric"]):
            return np.nan
        return apply_sensor_parameters(row["raw_numeric"], sensor_info)
    
    result_df["processed_value"] = result_df.apply(apply_params, axis=1)
    
    # Keep original raw sensor value for visualization
    result_df["sensor_raw_value"] = result_df["processed_value"]
    
    # Initialize calibration table
    calibration = CalibrationTable(sensor_info.calibration_data)
    
    if calibration.is_valid:
        # Apply calibration
        valid_mask = ~result_df["processed_value"].isna()
        valid_values = result_df.loc[valid_mask, "processed_value"].values
        
        liters, extrapolated = calibration.convert_array(valid_values)
        
        result_df["fuel_level_l_raw"] = np.nan
        result_df.loc[valid_mask, "fuel_level_l_raw"] = liters
        result_df["calibration_used"] = valid_mask & True
        result_df["calibration_points"] = calibration.num_points
        result_df["calibration_extrapolated"] = False
        result_df.loc[valid_mask, "calibration_extrapolated"] = extrapolated
    else:
        # Fallback: use processed value as-is
        result_df["fuel_level_l_raw"] = result_df["processed_value"]
        result_df["calibration_used"] = False
        result_df["calibration_points"] = 0
        result_df["calibration_extrapolated"] = False
    
    # Clean up intermediate columns (keep sensor_raw_value)
    result_df = result_df.drop(columns=["raw_numeric", "processed_value"])
    
    return result_df


def normalize_fuel_level(
    fuel_level_l: float,
    tank_volume: Optional[float]
) -> Optional[float]:
    """
    Normalize fuel level to 0-1 range based on tank volume.
    
    Args:
        fuel_level_l: Fuel level in liters
        tank_volume: Tank volume in liters
    
    Returns:
        Normalized value (0-1) or None if tank volume not available
    """
    if tank_volume is None or tank_volume <= 0:
        return None
    
    return min(max(fuel_level_l / tank_volume, 0.0), 1.0)



