"""
Event detection module
Detects fuel refuel and drain events with clustering
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class CandidatePoint:
    """A candidate point for event detection"""
    idx: int
    timestamp: datetime
    delta_l: float
    fuel_level: float
    lat: Optional[float] = None
    lng: Optional[float] = None
    speed_kmh: Optional[float] = None
    event_type: str = ""  # "refuel" or "drain"


@dataclass
class FuelEventCluster:
    """A cluster of candidate points forming an event"""
    event_type: str  # "refuel" or "drain"
    start_idx: int
    end_idx: int
    start_datetime: datetime
    end_datetime: datetime
    start_lat: Optional[float]
    start_lng: Optional[float]
    end_lat: Optional[float]
    end_lng: Optional[float]
    volume_change_l: float  # Absolute value
    signed_volume_l: float  # Positive for refuel, negative for drain
    samples_in_event: int
    max_gap_sec: int
    candidate_points: List[CandidatePoint] = field(default_factory=list)
    confidence: float = 0.0  # 0-1 confidence score


def calculate_thresholds(
    current_level: float,
    tank_volume: Optional[float],
    min_step_l: float,
    min_step_pct_tank: float,
    min_step_pct_level: float,
    mode: str = "absolute_and_relative"
) -> float:
    """
    Calculate detection threshold based on configuration.
    
    Args:
        current_level: Current fuel level in liters
        tank_volume: Tank volume in liters (optional)
        min_step_l: Absolute minimum step in liters
        min_step_pct_tank: Minimum step as percentage of tank
        min_step_pct_level: Minimum step as percentage of current level
        mode: "absolute_only", "relative_only", "absolute_and_relative"
    
    Returns:
        Threshold value in liters
    """
    thresholds = []
    
    if mode in ["absolute_only", "absolute_and_relative"]:
        thresholds.append(min_step_l)
    
    if mode in ["relative_only", "absolute_and_relative"]:
        if tank_volume and tank_volume > 0:
            thresholds.append(tank_volume * min_step_pct_tank / 100.0)
        
        if current_level and current_level > 0:
            thresholds.append(current_level * min_step_pct_level / 100.0)
    
    if not thresholds:
        return min_step_l
    
    return max(thresholds)


def detect_candidate_points(
    df: pd.DataFrame,
    value_column: str = "fuel_level_l",
    time_column: str = "device_time",
    tank_volume: Optional[float] = None,
    refuel_config: dict = None,
    drain_config: dict = None,
    mode: str = "absolute_and_relative"
) -> List[CandidatePoint]:
    """
    Detect candidate points for refuel/drain events.
    
    Uses delta-based detection with configurable thresholds.
    
    Args:
        df: DataFrame with fuel data
        value_column: Column with smoothed fuel values
        time_column: Column with timestamps
        tank_volume: Tank volume for relative thresholds
        refuel_config: Refuel detection thresholds
        drain_config: Drain detection thresholds
        mode: Threshold calculation mode
    
    Returns:
        List of candidate points
    """
    refuel_config = refuel_config or {
        "min_step_l": 5.0,
        "min_step_pct_tank": 1.0,
        "min_step_pct_level": 5.0
    }
    drain_config = drain_config or {
        "min_step_l": 5.0,
        "min_step_pct_tank": 1.0,
        "min_step_pct_level": 5.0
    }
    
    candidates = []
    values = df[value_column].values
    timestamps = pd.to_datetime(df[time_column])
    
    # Get optional columns
    lat_col = "lat" if "lat" in df.columns else None
    lng_col = "lng" if "lng" in df.columns else None
    speed_col = "speed_kmh" if "speed_kmh" in df.columns else None
    
    for i in range(1, len(values)):
        if pd.isna(values[i]) or pd.isna(values[i-1]):
            continue
        
        delta = values[i] - values[i-1]
        current_level = values[i-1]
        
        # Calculate thresholds
        refuel_threshold = calculate_thresholds(
            current_level, tank_volume,
            refuel_config["min_step_l"],
            refuel_config["min_step_pct_tank"],
            refuel_config["min_step_pct_level"],
            mode
        )
        
        drain_threshold = calculate_thresholds(
            current_level, tank_volume,
            drain_config["min_step_l"],
            drain_config["min_step_pct_tank"],
            drain_config["min_step_pct_level"],
            mode
        )
        
        event_type = None
        
        if delta >= refuel_threshold:
            event_type = "refuel"
        elif delta <= -drain_threshold:
            event_type = "drain"
        
        if event_type:
            candidates.append(CandidatePoint(
                idx=i,
                timestamp=timestamps.iloc[i],
                delta_l=delta,
                fuel_level=values[i],
                lat=df.iloc[i][lat_col] if lat_col else None,
                lng=df.iloc[i][lng_col] if lng_col else None,
                speed_kmh=df.iloc[i][speed_col] if speed_col else None,
                event_type=event_type
            ))
    
    return candidates


def cluster_candidate_points(
    candidates: List[CandidatePoint],
    merge_window_min: float = 15,
    min_points_in_cluster: int = 1,
    allow_mixed_sign: bool = False
) -> List[FuelEventCluster]:
    """
    Cluster nearby candidate points into events.
    
    Args:
        candidates: List of candidate points
        merge_window_min: Maximum time gap (minutes) to merge points
        min_points_in_cluster: Minimum points to form valid cluster
        allow_mixed_sign: Whether to allow mixed refuel/drain in same cluster
    
    Returns:
        List of event clusters
    """
    if not candidates:
        return []
    
    # Sort by timestamp
    sorted_candidates = sorted(candidates, key=lambda x: x.timestamp)
    
    clusters = []
    current_cluster_points = [sorted_candidates[0]]
    
    for i in range(1, len(sorted_candidates)):
        current_point = sorted_candidates[i]
        prev_point = current_cluster_points[-1]
        
        # Calculate time gap
        gap = (current_point.timestamp - prev_point.timestamp).total_seconds() / 60
        
        # Check if should merge
        should_merge = gap <= merge_window_min
        
        if not allow_mixed_sign:
            # Check if same event type
            should_merge = should_merge and (current_point.event_type == current_cluster_points[0].event_type)
        
        if should_merge:
            current_cluster_points.append(current_point)
        else:
            # Finalize current cluster
            if len(current_cluster_points) >= min_points_in_cluster:
                cluster = _create_cluster(current_cluster_points, allow_mixed_sign)
                if cluster:
                    clusters.append(cluster)
            
            # Start new cluster
            current_cluster_points = [current_point]
    
    # Handle last cluster
    if len(current_cluster_points) >= min_points_in_cluster:
        cluster = _create_cluster(current_cluster_points, allow_mixed_sign)
        if cluster:
            clusters.append(cluster)
    
    return clusters


def _create_cluster(
    points: List[CandidatePoint],
    allow_mixed_sign: bool
) -> Optional[FuelEventCluster]:
    """Create a cluster from candidate points"""
    if not points:
        return None
    
    # Calculate signed volume (sum of deltas)
    signed_volume = sum(p.delta_l for p in points)
    
    # Determine event type
    if allow_mixed_sign:
        event_type = "refuel" if signed_volume >= 0 else "drain"
    else:
        event_type = points[0].event_type
    
    # Calculate max gap
    gaps = []
    for i in range(1, len(points)):
        gap = (points[i].timestamp - points[i-1].timestamp).total_seconds()
        gaps.append(gap)
    max_gap = int(max(gaps)) if gaps else 0
    
    return FuelEventCluster(
        event_type=event_type,
        start_idx=points[0].idx,
        end_idx=points[-1].idx,
        start_datetime=points[0].timestamp,
        end_datetime=points[-1].timestamp,
        start_lat=points[0].lat,
        start_lng=points[0].lng,
        end_lat=points[-1].lat,
        end_lng=points[-1].lng,
        volume_change_l=abs(signed_volume),
        signed_volume_l=signed_volume,
        samples_in_event=len(points),
        max_gap_sec=max_gap,
        candidate_points=points,
        confidence=0.0  # Will be set by context validation
    )


def validate_events_with_context(
    clusters: List[FuelEventCluster],
    df: pd.DataFrame,
    context_config: dict = None,
    time_column: str = "device_time",
    speed_column: str = "speed_kmh",
    ignition_column: str = "ignition"
) -> List[FuelEventCluster]:
    """
    Validate events using contextual information.
    
    Checks:
    - Vehicle was stopped during event
    - Valid GPS quality
    - Ignition state
    
    Args:
        clusters: List of event clusters
        df: Original DataFrame with context columns
        context_config: Context validation configuration
        time_column: Timestamp column
        speed_column: Speed column
        ignition_column: Ignition column
    
    Returns:
        Validated clusters with confidence scores
    """
    context_config = context_config or {
        "max_valid_speed_kmh": 5.0,
        "max_hdop": 5.0,
        "use_ignition": True,
        "min_stop_duration_min": 2
    }
    
    validated = []
    timestamps = pd.to_datetime(df[time_column])
    
    for cluster in clusters:
        confidence = 1.0
        
        # Get data during event window
        event_mask = (timestamps >= cluster.start_datetime) & (timestamps <= cluster.end_datetime)
        event_data = df[event_mask]
        
        if event_data.empty:
            cluster.confidence = 0.5
            validated.append(cluster)
            continue
        
        # Check speed (vehicle should be stopped)
        if speed_column in df.columns:
            avg_speed = event_data[speed_column].mean()
            if pd.notna(avg_speed):
                if avg_speed <= context_config["max_valid_speed_kmh"]:
                    confidence *= 1.0  # Good - vehicle stopped
                elif avg_speed <= context_config["max_valid_speed_kmh"] * 2:
                    confidence *= 0.7  # Suspicious - slow movement
                else:
                    confidence *= 0.3  # Unlikely - vehicle moving
        
        # Check ignition (for refuels, typically ignition is off)
        if context_config["use_ignition"] and ignition_column in df.columns:
            ign_values = event_data[ignition_column].fillna(True)
            if cluster.event_type == "refuel":
                # Refuels typically happen with ignition off
                pct_off = (ign_values == False).mean()
                confidence *= (0.7 + 0.3 * pct_off)
        
        # Event duration check
        duration_min = (cluster.end_datetime - cluster.start_datetime).total_seconds() / 60
        if duration_min < context_config["min_stop_duration_min"]:
            confidence *= 0.8  # Very quick event
        
        cluster.confidence = min(max(confidence, 0.0), 1.0)
        validated.append(cluster)
    
    return validated


def filter_events_by_volume(
    clusters: List[FuelEventCluster],
    tank_volume: Optional[float],
    min_volume_l: float = 10.0,
    min_volume_pct_tank: float = 2.0
) -> List[FuelEventCluster]:
    """
    Filter events by minimum volume threshold.
    
    Args:
        clusters: List of event clusters
        tank_volume: Tank volume for percentage calculation
        min_volume_l: Minimum absolute volume in liters
        min_volume_pct_tank: Minimum volume as percentage of tank
    
    Returns:
        Filtered list of clusters
    """
    filtered = []
    
    for cluster in clusters:
        # Calculate threshold
        threshold = min_volume_l
        
        if tank_volume and tank_volume > 0:
            pct_threshold = tank_volume * min_volume_pct_tank / 100.0
            threshold = max(threshold, pct_threshold)
        
        if cluster.volume_change_l >= threshold:
            filtered.append(cluster)
    
    return filtered


def detect_fuel_events(
    df: pd.DataFrame,
    config: dict,
    value_column: str = "fuel_level_l",
    time_column: str = "device_time",
    tank_volume: Optional[float] = None
) -> List[FuelEventCluster]:
    """
    Main function to detect fuel events.
    
    Pipeline:
    1. Detect candidate points
    2. Cluster candidates
    3. Validate with context
    4. Filter by minimum volume
    
    Args:
        df: DataFrame with smoothed fuel data
        config: Processing configuration
        value_column: Column with fuel values
        time_column: Column with timestamps
        tank_volume: Tank volume in liters
    
    Returns:
        List of detected fuel events
    """
    detection_cfg = config.get("detection", {})
    context_cfg = config.get("context", {})
    
    # Step 1: Detect candidates
    candidates = detect_candidate_points(
        df=df,
        value_column=value_column,
        time_column=time_column,
        tank_volume=tank_volume,
        refuel_config=detection_cfg.get("refuel", {}),
        drain_config=detection_cfg.get("drain", {}),
        mode=detection_cfg.get("mode", "absolute_and_relative")
    )
    
    if not candidates:
        return []
    
    # Step 2: Cluster candidates
    cluster_cfg = detection_cfg.get("cluster", {})
    
    # Calculate adaptive merge window
    smoothing_level = config.get("smoothing", {}).get("smoothing_level", 0.5)
    merge_window_min = cluster_cfg.get("merge_window_min_min", 5) + \
                       (cluster_cfg.get("merge_window_min_max", 30) - cluster_cfg.get("merge_window_min_min", 5)) * smoothing_level
    
    clusters = cluster_candidate_points(
        candidates=candidates,
        merge_window_min=merge_window_min,
        min_points_in_cluster=cluster_cfg.get("min_points_in_cluster", 1),
        allow_mixed_sign=detection_cfg.get("event", {}).get("allow_mixed_sign_cluster", False)
    )
    
    # Step 3: Validate with context
    clusters = validate_events_with_context(
        clusters=clusters,
        df=df,
        context_config=context_cfg,
        time_column=time_column
    )
    
    # Step 4: Filter by volume
    event_cfg = detection_cfg.get("event", {})
    clusters = filter_events_by_volume(
        clusters=clusters,
        tank_volume=tank_volume,
        min_volume_l=event_cfg.get("min_volume_l", 10.0),
        min_volume_pct_tank=event_cfg.get("min_volume_pct_tank", 2.0)
    )
    
    return clusters


def events_to_dataframe(
    events: List[FuelEventCluster],
    device_id: int,
    object_id: Optional[int],
    vehicle_id: Optional[int],
    processing_version: str,
    processing_params: dict
) -> pd.DataFrame:
    """
    Convert events to DataFrame matching silver_data_layer.fuel_events schema.
    """
    if not events:
        return pd.DataFrame()
    
    records = []
    for event in events:
        records.append({
            "device_id": device_id,
            "object_id": object_id,
            "vehicle_id": vehicle_id,
            "event_type": event.event_type,
            "start_datetime": event.start_datetime,
            "end_datetime": event.end_datetime,
            "start_lat": event.start_lat,
            "start_lng": event.start_lng,
            "end_lat": event.end_lat,
            "end_lng": event.end_lng,
            "volume_change_l": event.volume_change_l,
            "signed_volume_l": event.signed_volume_l,
            "samples_in_event": event.samples_in_event,
            "max_gap_sec": event.max_gap_sec,
            "confidence": event.confidence,
            "processing_version": processing_version,
            "processing_params": processing_params
        })
    
    return pd.DataFrame(records)


