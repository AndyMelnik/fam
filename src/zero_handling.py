"""
Zero value handling module (v2)
Handles "sticky zeros" and distinguishes valid vs invalid zero readings
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import timedelta


@dataclass
class ZeroHandlingResult:
    """Result of zero handling"""
    zeros_removed: int
    zeros_kept: int
    grace_period_points: int
    zero_plateaus_found: int


def detect_ignition_events(
    df: pd.DataFrame,
    ignition_column: str = "ignition",
    time_column: str = "device_time"
) -> List[pd.Timestamp]:
    """
    Detect ignition ON events (transitions from OFF to ON).
    
    Returns:
        List of timestamps when ignition turned ON
    """
    if ignition_column not in df.columns:
        return []
    
    # Ensure boolean
    ign = df[ignition_column].fillna(False).astype(bool)
    
    # Find transitions (0->1)
    transitions = ign.astype(int).diff() == 1
    
    return df.loc[transitions, time_column].tolist()


def detect_zero_runs(
    values: np.ndarray,
    timestamps: pd.Series,
    max_zero_level: float = 2.0,
    min_run_minutes: int = 10
) -> List[Tuple[int, int, bool]]:
    """
    Detect continuous runs of zero/near-zero values.
    
    Args:
        values: Fuel level values
        timestamps: Corresponding timestamps
        max_zero_level: Maximum value to consider as "zero"
        min_run_minutes: Minimum duration to consider a valid zero plateau
    
    Returns:
        List of (start_idx, end_idx, is_valid_plateau) tuples
    """
    n = len(values)
    if n == 0:
        return []
    
    runs = []
    is_zero = values <= max_zero_level
    
    start_idx = None
    
    for i in range(n):
        if is_zero[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                end_idx = i - 1
                
                # Calculate run duration
                if isinstance(timestamps.iloc[0], pd.Timestamp):
                    duration = (timestamps.iloc[end_idx] - timestamps.iloc[start_idx]).total_seconds() / 60
                else:
                    duration = (timestamps.iloc[end_idx] - timestamps.iloc[start_idx]) / 60
                
                is_valid = duration >= min_run_minutes
                runs.append((start_idx, end_idx, is_valid))
                start_idx = None
    
    # Handle run at end
    if start_idx is not None:
        end_idx = n - 1
        if isinstance(timestamps.iloc[0], pd.Timestamp):
            duration = (timestamps.iloc[end_idx] - timestamps.iloc[start_idx]).total_seconds() / 60
        else:
            duration = (timestamps.iloc[end_idx] - timestamps.iloc[start_idx]) / 60
        
        is_valid = duration >= min_run_minutes
        runs.append((start_idx, end_idx, is_valid))
    
    return runs


def handle_zeros_v2(
    df: pd.DataFrame,
    value_column: str = "fuel_level_l_raw",
    time_column: str = "device_time",
    ignition_column: str = "ignition",
    speed_column: str = "speed_kmh",
    mode: str = "auto",
    startup_grace_period_min: int = 5,
    min_zero_run_min: int = 10,
    max_zero_level_l: float = 2.0
) -> Tuple[pd.DataFrame, ZeroHandlingResult]:
    """
    Handle zero values in fuel data.
    
    Logic:
    1. Grace period after ignition ON: ignore zeros for X minutes
    2. Stable zero plateaus at stops: consider valid (empty tank)
    3. Sporadic zeros during movement: mark as invalid
    
    Args:
        df: DataFrame with fuel data
        value_column: Column with fuel level values
        time_column: Column with timestamps
        ignition_column: Column with ignition state
        speed_column: Column with vehicle speed
        mode: "auto", "keep", "interpolate", "drop"
        startup_grace_period_min: Minutes after ignition ON to ignore zeros
        min_zero_run_min: Minimum duration for valid zero plateau
        max_zero_level_l: Maximum value to consider as "zero"
    
    Returns:
        Tuple of (processed DataFrame, stats)
    """
    result_df = df.copy()
    result_df["zero_handling_flag"] = "valid"  # valid, grace_period, invalid_zero, interpolated
    
    if mode == "keep":
        return result_df, ZeroHandlingResult(0, 0, 0, 0)
    
    values = result_df[value_column].values.copy()
    timestamps = pd.to_datetime(result_df[time_column])
    
    # Identify all zeros
    is_zero = values <= max_zero_level_l
    zeros_total = is_zero.sum()
    
    if zeros_total == 0:
        return result_df, ZeroHandlingResult(0, 0, 0, 0)
    
    # Grace period after ignition events
    grace_period_mask = np.zeros(len(df), dtype=bool)
    
    if ignition_column in df.columns:
        ignition_events = detect_ignition_events(df, ignition_column, time_column)
        
        for ign_time in ignition_events:
            grace_end = ign_time + timedelta(minutes=startup_grace_period_min)
            period_mask = (timestamps >= ign_time) & (timestamps <= grace_end)
            grace_period_mask |= period_mask.values
    
    # Detect zero runs
    zero_runs = detect_zero_runs(
        values,
        timestamps,
        max_zero_level=max_zero_level_l,
        min_run_minutes=min_zero_run_min
    )
    
    # Build valid zero mask (long stable plateaus are valid)
    valid_zero_mask = np.zeros(len(df), dtype=bool)
    for start_idx, end_idx, is_valid in zero_runs:
        if is_valid:
            valid_zero_mask[start_idx:end_idx+1] = True
    
    # Movement-based validation
    if speed_column in df.columns:
        speeds = result_df[speed_column].fillna(0).values
        is_moving = speeds > 5.0  # km/h threshold
    else:
        is_moving = np.zeros(len(df), dtype=bool)
    
    # Apply logic
    zeros_removed = 0
    zeros_kept = 0
    grace_period_points = 0
    
    for i in range(len(df)):
        if not is_zero[i]:
            continue
        
        # Check grace period
        if grace_period_mask[i]:
            result_df.loc[result_df.index[i], "zero_handling_flag"] = "grace_period"
            grace_period_points += 1
            
            if mode in ["auto", "interpolate"]:
                values[i] = np.nan  # Will be interpolated later
                zeros_removed += 1
            else:
                zeros_kept += 1
            continue
        
        # Check if valid zero (stable plateau at stop)
        if valid_zero_mask[i] and not is_moving[i]:
            zeros_kept += 1
            continue
        
        # Sporadic zero during movement = invalid
        if is_moving[i]:
            result_df.loc[result_df.index[i], "zero_handling_flag"] = "invalid_zero"
            
            if mode in ["auto", "interpolate"]:
                values[i] = np.nan
                zeros_removed += 1
            elif mode == "drop":
                values[i] = np.nan
                zeros_removed += 1
            else:
                zeros_kept += 1
        else:
            # Short zero at stop - keep but flag
            zeros_kept += 1
    
    # Interpolate NaN values
    if mode in ["auto", "interpolate"]:
        series = pd.Series(values)
        interpolated = series.interpolate(method='linear', limit_direction='both')
        values = interpolated.values
        
        # Update flag for interpolated values
        interp_mask = result_df["zero_handling_flag"].isin(["grace_period", "invalid_zero"])
        result_df.loc[interp_mask, "zero_handling_flag"] = "interpolated"
    
    result_df[value_column] = values
    
    stats = ZeroHandlingResult(
        zeros_removed=zeros_removed,
        zeros_kept=zeros_kept,
        grace_period_points=grace_period_points,
        zero_plateaus_found=len([r for r in zero_runs if r[2]])
    )
    
    return result_df, stats



