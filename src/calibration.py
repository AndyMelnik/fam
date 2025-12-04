"""
Calibration and unit conversion module
Converts raw sensor values to calibrated output using calibration tables.
Supports different measurement units (liters, gallons, percent, etc.)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from .database import SensorInfo


# Known measurement unit types
VOLUME_UNITS = ["l", "liters", "litre", "litres", "liter", "л", "литры", "литр"]
GALLON_UNITS = ["gal", "gallon", "gallons", "галлон", "галлоны"]
PERCENT_UNITS = ["%", "percent", "pct", "процент", "проценты"]


def normalize_units(sensor_units: Optional[str]) -> str:
    """
    Normalize sensor units to a standard display format.
    
    Args:
        sensor_units: Raw sensor units from database
        
    Returns:
        Normalized unit string for display
    """
    if sensor_units is None:
        return "L"  # Default to liters
    
    units_lower = sensor_units.lower().strip()
    
    if units_lower in VOLUME_UNITS:
        return "L"
    elif units_lower in GALLON_UNITS:
        return "gal"
    elif units_lower in PERCENT_UNITS:
        return "%"
    else:
        # Return as-is if unknown
        return sensor_units


def get_unit_description(sensor_units: Optional[str]) -> str:
    """
    Get a human-readable description of the measurement units.
    """
    if sensor_units is None:
        return "Liters (default)"
    
    units_lower = sensor_units.lower().strip()
    
    if units_lower in VOLUME_UNITS:
        return "Liters"
    elif units_lower in GALLON_UNITS:
        return "Gallons"
    elif units_lower in PERCENT_UNITS:
        return "Percent of tank"
    else:
        return sensor_units


@dataclass
class CalibrationResult:
    """Result of calibration conversion"""
    value: float  # Calibrated value in sensor units
    calibration_used: bool
    calibration_points: int
    interpolation_type: str  # "exact", "interpolated", "clipped_low", "clipped_high", "fallback"
    units: str = "L"  # Measurement units


class CalibrationTable:
    """
    Calibration table for converting raw sensor values to calibrated output.
    Uses linear interpolation within range, clips values outside range.
    """
    
    def __init__(
        self, 
        calibration_data: Optional[List[Dict[str, float]]] = None,
        sensor_units: Optional[str] = None
    ):
        """
        Initialize calibration table.
        
        Args:
            calibration_data: List of {"in": raw_value, "out": calibrated_value} dictionaries
            sensor_units: Measurement units from sensor description
        """
        self.points: List[Tuple[float, float]] = []
        self._in_values: np.ndarray = np.array([])
        self._out_values: np.ndarray = np.array([])
        self.units = normalize_units(sensor_units)
        self.units_description = get_unit_description(sensor_units)
        
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
        Convert raw sensor value using calibration table.
        Values outside calibration range are clipped to boundary values.
        
        Args:
            raw_value: Raw sensor reading
        
        Returns:
            CalibrationResult with converted value and metadata
        """
        if not self.is_valid:
            return CalibrationResult(
                value=raw_value,
                calibration_used=False,
                calibration_points=0,
                interpolation_type="fallback",
                units=self.units
            )
        
        min_in, max_in = self.range
        
        # Check for exact match
        exact_idx = np.where(np.isclose(self._in_values, raw_value, rtol=1e-6))[0]
        if len(exact_idx) > 0:
            return CalibrationResult(
                value=float(self._out_values[exact_idx[0]]),
                calibration_used=True,
                calibration_points=self.num_points,
                interpolation_type="exact",
                units=self.units
            )
        
        # Handle values outside calibration range - CLIP instead of extrapolate
        if raw_value < min_in:
            # Clip to minimum calibration value
            return CalibrationResult(
                value=float(self._out_values[0]),  # Use the minimum output value
                calibration_used=True,
                calibration_points=self.num_points,
                interpolation_type="clipped_low",
                units=self.units
            )
        elif raw_value > max_in:
            # Clip to maximum calibration value
            return CalibrationResult(
                value=float(self._out_values[-1]),  # Use the maximum output value
                calibration_used=True,
                calibration_points=self.num_points,
                interpolation_type="clipped_high",
                units=self.units
            )
        
        # Linear interpolation within range
        result = np.interp(raw_value, self._in_values, self._out_values)
        
        return CalibrationResult(
            value=float(result),
            calibration_used=True,
            calibration_points=self.num_points,
            interpolation_type="interpolated",
            units=self.units
        )
    
    def convert_array(self, raw_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert array of raw values using calibration table.
        Values outside calibration range are CLIPPED to boundary values.
        
        Returns:
            Tuple of (calibrated_array, clipped_low_mask, clipped_high_mask)
        """
        if not self.is_valid:
            # Return raw values as-is, mark all as not clipped
            return raw_values.copy(), np.zeros(len(raw_values), dtype=bool), np.zeros(len(raw_values), dtype=bool)
        
        # Clip raw values to calibration range BEFORE interpolation
        min_in, max_in = self.range
        clipped_low = raw_values < min_in
        clipped_high = raw_values > max_in
        
        # Clip raw values to calibration boundaries
        clipped_raw = np.clip(raw_values, min_in, max_in)
        
        # Now interpolate - all values are within range
        calibrated = np.interp(clipped_raw, self._in_values, self._out_values)
        
        return calibrated, clipped_low, clipped_high
    
    @property
    def output_range(self) -> Tuple[float, float]:
        """Get output range (min_out, max_out) from calibration table"""
        if not self.is_valid:
            return (0.0, 0.0)
        return (float(self._out_values.min()), float(self._out_values.max()))


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


def convert_to_calibrated(
    df: pd.DataFrame,
    sensor_info: SensorInfo,
    value_column: str = "value"
) -> Tuple[pd.DataFrame, str, str]:
    """
    Convert raw sensor data using calibration table.
    Values outside calibration range are CLIPPED to boundary values.
    
    Args:
        df: DataFrame with raw sensor values
        sensor_info: Sensor configuration with calibration data
        value_column: Name of column with raw values
    
    Returns:
        Tuple of (DataFrame, units_short, units_description) where DataFrame has columns:
            - sensor_raw_value: Original raw sensor value (before calibration)
            - fuel_level_l_raw: Calibrated value in measurement units
            - calibration_used: Whether calibration was applied
            - calibration_points: Number of calibration points used
            - calibration_clipped_low: Value was below calibration range (clipped)
            - calibration_clipped_high: Value was above calibration range (clipped)
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
    
    # Initialize calibration table with units
    calibration = CalibrationTable(
        sensor_info.calibration_data,
        sensor_info.sensor_units
    )
    
    # Get units info
    units_short = calibration.units
    units_description = calibration.units_description
    
    if calibration.is_valid:
        # Apply calibration with clipping
        valid_mask = ~result_df["processed_value"].isna()
        valid_values = result_df.loc[valid_mask, "processed_value"].values
        
        calibrated, clipped_low, clipped_high = calibration.convert_array(valid_values)
        
        result_df["fuel_level_l_raw"] = np.nan
        result_df.loc[valid_mask, "fuel_level_l_raw"] = calibrated
        result_df["calibration_used"] = valid_mask & True
        result_df["calibration_points"] = calibration.num_points
        result_df["calibration_clipped_low"] = False
        result_df["calibration_clipped_high"] = False
        result_df.loc[valid_mask, "calibration_clipped_low"] = clipped_low
        result_df.loc[valid_mask, "calibration_clipped_high"] = clipped_high
        # Keep backward compatibility column
        result_df["calibration_extrapolated"] = result_df["calibration_clipped_low"] | result_df["calibration_clipped_high"]
    else:
        # Fallback: use processed value as-is (no calibration)
        result_df["fuel_level_l_raw"] = result_df["processed_value"]
        result_df["calibration_used"] = False
        result_df["calibration_points"] = 0
        result_df["calibration_clipped_low"] = False
        result_df["calibration_clipped_high"] = False
        result_df["calibration_extrapolated"] = False
    
    # Clean up intermediate columns (keep sensor_raw_value)
    result_df = result_df.drop(columns=["raw_numeric", "processed_value"])
    
    return result_df, units_short, units_description


# Backward compatibility alias
def convert_to_liters(
    df: pd.DataFrame,
    sensor_info: SensorInfo,
    value_column: str = "value"
) -> pd.DataFrame:
    """
    Backward compatible function - converts raw sensor data using calibration.
    Values outside calibration range are CLIPPED.
    
    Returns:
        DataFrame with calibrated values
    """
    result_df, _, _ = convert_to_calibrated(df, sensor_info, value_column)
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



