"""
Smoothing algorithms module
Implements Hampel filter, Rolling Median, and EMA
Full pipeline: Hampel → Median → EMA
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class SmoothingStats:
    """Statistics from smoothing operation"""
    outliers_removed: int
    total_points: int
    median_window_used: int
    ema_alpha_used: float
    hampel_window_sec_used: float
    hampel_sigma_used: float


def hampel_filter(
    values: np.ndarray,
    timestamps: np.ndarray,
    window_sec: float = 300,
    n_sigma: float = 3.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hampel filter for outlier detection and removal.
    
    Uses Median Absolute Deviation (MAD) for robust outlier detection.
    Time-based window for irregular time series.
    
    Args:
        values: Array of values to filter
        timestamps: Array of timestamps (datetime or numeric seconds)
        window_sec: Window size in seconds
        n_sigma: Number of sigma for outlier threshold (using MAD)
    
    Returns:
        Tuple of (filtered_values, outlier_mask)
    """
    n = len(values)
    filtered = values.copy()
    outlier_mask = np.zeros(n, dtype=bool)
    
    if n < 3:
        return filtered, outlier_mask
    
    # Convert timestamps to seconds if datetime
    if isinstance(timestamps[0], (pd.Timestamp, np.datetime64)):
        ts_seconds = (pd.to_datetime(timestamps) - pd.to_datetime(timestamps[0])).total_seconds()
        if hasattr(ts_seconds, 'values'):
            ts_seconds = ts_seconds.values
        else:
            ts_seconds = np.array([t.total_seconds() for t in (pd.to_datetime(timestamps) - pd.to_datetime(timestamps[0]))])
    else:
        ts_seconds = np.array(timestamps)
    
    # MAD scale factor for normal distribution
    k = 1.4826
    
    for i in range(n):
        # Find points within time window
        half_window = window_sec / 2
        t_center = ts_seconds[i]
        
        window_mask = (ts_seconds >= t_center - half_window) & (ts_seconds <= t_center + half_window)
        window_values = values[window_mask]
        
        if len(window_values) < 3:
            continue
        
        # Calculate robust statistics
        median = np.median(window_values)
        mad = np.median(np.abs(window_values - median))
        
        # Robust sigma estimate
        sigma = k * mad
        
        if sigma == 0:
            sigma = np.std(window_values)
        
        if sigma == 0:
            continue
        
        # Check if current point is outlier
        if np.abs(values[i] - median) > n_sigma * sigma:
            filtered[i] = median
            outlier_mask[i] = True
    
    return filtered, outlier_mask


def rolling_median(
    values: np.ndarray,
    window_size: int = 5,
    min_periods: int = 1
) -> np.ndarray:
    """
    Rolling median filter.
    
    Args:
        values: Array of values to smooth
        window_size: Number of points in window
        min_periods: Minimum periods required for valid result
    
    Returns:
        Smoothed values
    """
    if len(values) < min_periods:
        return values.copy()
    
    # Use pandas for efficient rolling median
    series = pd.Series(values)
    smoothed = series.rolling(
        window=window_size,
        min_periods=min_periods,
        center=True
    ).median()
    
    # Fill NaN at edges
    smoothed = smoothed.bfill().ffill()
    
    return smoothed.values


def exponential_moving_average(
    values: np.ndarray,
    alpha: float = 0.3
) -> np.ndarray:
    """
    Exponential Moving Average (EMA) filter.
    
    Args:
        values: Array of values to smooth
        alpha: Smoothing factor (0-1). Lower = more smoothing.
    
    Returns:
        Smoothed values
    """
    if len(values) == 0:
        return values.copy()
    
    # Use pandas ewm for efficient calculation
    series = pd.Series(values)
    smoothed = series.ewm(alpha=alpha, adjust=False).mean()
    
    return smoothed.values


def apply_smoothing(
    values: np.ndarray,
    timestamps: np.ndarray,
    smoothing_level: int = 5,
    hampel_window_min_sec: float = 60,
    hampel_window_max_sec: float = 900,
    hampel_sigma_min: float = 2.0,
    hampel_sigma_max: float = 4.0,
    median_window_min_points: int = 3,
    median_window_max_points: int = 15,
    ema_alpha_min: float = 0.05,
    ema_alpha_max: float = 0.5
) -> Tuple[np.ndarray, SmoothingStats]:
    """
    Apply full smoothing pipeline: Hampel → Median → EMA
    
    Parameters are automatically calculated based on smoothing_level (1-10):
    - Level 1: Minimal smoothing (wide sigma, small windows, high alpha)
    - Level 10: Maximum smoothing (tight sigma, large windows, low alpha)
    
    Args:
        values: Raw values to smooth
        timestamps: Corresponding timestamps
        smoothing_level: Smoothing intensity from 1 to 10
        *_min/*_max: Parameter ranges for adaptive calculation
    
    Returns:
        Tuple of (smoothed_values, stats)
    """
    # Normalize smoothing_level to 0-1 range
    level = (smoothing_level - 1) / 9.0  # 1->0, 10->1
    level = max(0.0, min(1.0, level))
    
    # Interpolate parameters based on level
    def lerp(min_val: float, max_val: float, t: float) -> float:
        return min_val + (max_val - min_val) * t
    
    # Higher level = more smoothing
    hampel_window_sec = lerp(hampel_window_min_sec, hampel_window_max_sec, level)
    hampel_sigma = lerp(hampel_sigma_max, hampel_sigma_min, level)  # Inverted: lower sigma = more outlier removal
    median_window_points = int(lerp(median_window_min_points, median_window_max_points, level))
    ema_alpha = lerp(ema_alpha_max, ema_alpha_min, level)  # Inverted: lower alpha = more smoothing
    
    current = values.copy()
    outliers_removed = 0
    
    # Stage 1: Hampel filter (outlier removal)
    current, outlier_mask = hampel_filter(
        current,
        timestamps,
        window_sec=hampel_window_sec,
        n_sigma=hampel_sigma
    )
    outliers_removed = int(np.sum(outlier_mask))
    
    # Stage 2: Rolling Median (local smoothing)
    current = rolling_median(current, window_size=median_window_points)
    
    # Stage 3: EMA (global smoothing)
    current = exponential_moving_average(current, alpha=ema_alpha)
    
    stats = SmoothingStats(
        outliers_removed=outliers_removed,
        total_points=len(values),
        median_window_used=median_window_points,
        ema_alpha_used=ema_alpha,
        hampel_window_sec_used=hampel_window_sec,
        hampel_sigma_used=hampel_sigma
    )
    
    return current, stats


def apply_smoothing_to_dataframe(
    df: pd.DataFrame,
    config: dict,
    value_column: str = "fuel_level_l_raw",
    time_column: str = "device_time",
    output_column: str = "fuel_level_l"
) -> Tuple[pd.DataFrame, SmoothingStats]:
    """
    Apply smoothing to a DataFrame.
    
    Args:
        df: DataFrame with fuel data
        config: Processing configuration dict
        value_column: Column with raw values
        time_column: Column with timestamps
        output_column: Name for output column
    
    Returns:
        Tuple of (DataFrame with smoothed column, stats)
    """
    result_df = df.copy()
    
    # Extract valid values
    valid_mask = ~result_df[value_column].isna()
    
    if valid_mask.sum() < 3:
        result_df[output_column] = result_df[value_column]
        return result_df, SmoothingStats(0, len(df), 0, 0, 0, 0)
    
    values = result_df.loc[valid_mask, value_column].values
    timestamps = result_df.loc[valid_mask, time_column].values
    
    # Get smoothing parameters from config
    smoothing_cfg = config.get("smoothing", {})
    smoothing_level = smoothing_cfg.get("smoothing_level", 5)
    
    # Ensure smoothing_level is in 1-10 range
    if smoothing_level <= 1:
        # Might be old 0-1 format, convert to 1-10
        smoothing_level = int(smoothing_level * 9) + 1
    smoothing_level = max(1, min(10, int(smoothing_level)))
    
    hampel_cfg = smoothing_cfg.get("hampel", {})
    median_cfg = smoothing_cfg.get("median", {})
    ema_cfg = smoothing_cfg.get("ema", {})
    
    # Apply smoothing
    smoothed, stats = apply_smoothing(
        values=values,
        timestamps=timestamps,
        smoothing_level=smoothing_level,
        hampel_window_min_sec=hampel_cfg.get("window_min_sec", 60),
        hampel_window_max_sec=hampel_cfg.get("window_max_sec", 900),
        hampel_sigma_min=hampel_cfg.get("sigma_min", 2.0),
        hampel_sigma_max=hampel_cfg.get("sigma_max", 4.0),
        median_window_min_points=median_cfg.get("window_min_points", 3),
        median_window_max_points=median_cfg.get("window_max_points", 15),
        ema_alpha_min=ema_cfg.get("alpha_min", 0.05),
        ema_alpha_max=ema_cfg.get("alpha_max", 0.5)
    )
    
    # Apply smoothed values
    result_df[output_column] = np.nan
    result_df.loc[valid_mask, output_column] = smoothed
    
    return result_df, stats
