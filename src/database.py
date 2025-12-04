"""
Database connection and data extraction module
Handles Bronze Layer data extraction from PostgreSQL
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import json

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


@dataclass
class ObjectInfo:
    """Information about a telematics object"""
    object_id: int
    object_label: str
    device_id: int
    vehicle_id: Optional[int] = None
    vehicle_label: Optional[str] = None
    fuel_tank_volume: Optional[float] = None


@dataclass
class SensorInfo:
    """Information about a fuel sensor"""
    sensor_id: int
    device_id: int
    input_label: str
    sensor_label: str
    sensor_type: str
    multiplier: Optional[float] = None
    divider: Optional[float] = None
    accuracy: Optional[float] = None
    sensor_units: Optional[str] = None
    calibration_data: Optional[List[Dict[str, float]]] = None
    parameters: Optional[Dict[str, Any]] = None


class DatabaseConnector:
    """PostgreSQL database connector for Bronze Layer"""

    def __init__(self, connection_string: str):
        """
        Initialize database connection.
        
        Args:
            connection_string: PostgreSQL connection URL
                Example: postgresql://user:password@host:port/database
        """
        self.connection_string = connection_string
        self._engine: Optional[Engine] = None

    @property
    def engine(self) -> Engine:
        """Lazy engine initialization"""
        if self._engine is None:
            self._engine = create_engine(self.connection_string)
        return self._engine

    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def get_objects_with_fuel_sensors(self) -> List[ObjectInfo]:
        """
        Get list of objects that have fuel sensors configured.
        Returns only objects with sensor_type = 'fuel'.
        """
        query = """
        SELECT DISTINCT
            o.object_id,
            o.object_label,
            o.device_id,
            v.vehicle_id,
            v.vehicle_label,
            v.fuel_tank_volume
        FROM raw_business_data.objects o
        LEFT JOIN raw_business_data.vehicles v 
            ON v.object_id = o.object_id
        INNER JOIN raw_business_data.sensor_description sd
            ON sd.device_id = o.device_id
            AND LOWER(sd.sensor_type) = 'fuel'
        ORDER BY o.object_label
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        return [
            ObjectInfo(
                object_id=row["object_id"],
                object_label=row["object_label"],
                device_id=row["device_id"],
                vehicle_id=row["vehicle_id"],
                vehicle_label=row["vehicle_label"],
                fuel_tank_volume=row["fuel_tank_volume"]
            )
            for _, row in df.iterrows()
        ]

    def get_objects_with_sensors_details(self) -> pd.DataFrame:
        """
        Get detailed table of objects with their fuel sensor configurations.
        Returns DataFrame with: object_id, object_label, device_id, sensor_label, 
        input_label, sensor_type, calibration_data
        """
        query = """
        SELECT 
            o.object_id,
            o.object_label,
            o.device_id,
            sd.sensor_label,
            sd.input_label,
            sd.sensor_type,
            sd.calibration_data
        FROM raw_business_data.objects o
        INNER JOIN raw_business_data.sensor_description sd
            ON sd.device_id = o.device_id
            AND LOWER(sd.sensor_type) = 'fuel'
        ORDER BY o.object_label, sd.sensor_label
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        
        # Format calibration_data for display
        def format_calibration(cal_data):
            if cal_data is None:
                return "N/A"
            if isinstance(cal_data, str):
                try:
                    cal_data = json.loads(cal_data)
                except:
                    return str(cal_data)[:50] + "..."
            if isinstance(cal_data, list):
                if len(cal_data) == 0:
                    return "Empty"
                return f"{len(cal_data)} points"
            return str(cal_data)[:50] + "..."
        
        df["calibration_data"] = df["calibration_data"].apply(format_calibration)
        
        return df

    def get_fuel_sensors_for_device(self, device_id: int) -> List[SensorInfo]:
        """
        Get fuel sensors configuration for a specific device.
        """
        query = """
        SELECT 
            sd.sensor_id,
            sd.device_id,
            sd.input_label,
            sd.sensor_label,
            sd.sensor_type,
            sd.multiplier,
            sd.divider,
            sd.accuracy,
            sd.sensor_units,
            sd.calibration_data,
            sd.parameters
        FROM raw_business_data.sensor_description sd
        WHERE sd.device_id = :device_id
        AND sd.sensor_type = 'fuel'
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params={"device_id": device_id})
        
        sensors = []
        for _, row in df.iterrows():
            calibration = row["calibration_data"]
            if isinstance(calibration, str):
                calibration = json.loads(calibration)
            
            parameters = row["parameters"]
            if isinstance(parameters, str):
                parameters = json.loads(parameters)
            
            sensors.append(SensorInfo(
                sensor_id=row["sensor_id"],
                device_id=row["device_id"],
                input_label=row["input_label"],
                sensor_label=row["sensor_label"],
                sensor_type=row["sensor_type"],
                multiplier=row["multiplier"],
                divider=row["divider"],
                accuracy=row["accuracy"],
                sensor_units=row["sensor_units"],
                calibration_data=calibration,
                parameters=parameters
            ))
        
        return sensors

    def get_fuel_sensor_data(
        self,
        device_id: int,
        sensor_names: List[str],
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Get raw fuel sensor data from inputs table.
        
        Returns DataFrame with columns:
            - device_id
            - device_time
            - sensor_name
            - value (raw sensor value)
        """
        query = """
        SELECT 
            i.device_id,
            i.device_time AT TIME ZONE 'UTC' AS device_time,
            i.sensor_name,
            i.value
        FROM raw_telematics_data.inputs i
        WHERE i.device_id = :device_id
        AND i.sensor_name = ANY(:sensor_names)
        AND i.device_time >= :start_time
        AND i.device_time < :end_time
        ORDER BY i.device_time
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params={
                    "device_id": device_id,
                    "sensor_names": sensor_names,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )
        
        return df

    def get_gps_data(
        self,
        device_id: int,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Get GPS tracking data for a device.
        
        Returns DataFrame with columns:
            - device_id
            - device_time
            - lat (degrees)
            - lng (degrees)
            - speed_kmh
            - hdop
        """
        query = """
        SELECT 
            device_id,
            device_time AT TIME ZONE 'UTC' AS device_time,
            latitude / 1e7::float AS lat,
            longitude / 1e7::float AS lng,
            speed / 100.0::float AS speed_kmh,
            hdop
        FROM raw_telematics_data.tracking_data_core
        WHERE device_id = :device_id
        AND device_time >= :start_time
        AND device_time < :end_time
        ORDER BY device_time
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params={
                    "device_id": device_id,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )
        
        return df

    def get_ignition_states(
        self,
        device_id: int,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """
        Get ignition state data from states table.
        
        Returns DataFrame with columns:
            - device_id
            - device_time
            - ignition (boolean)
        """
        query = """
        SELECT 
            device_id,
            device_time AT TIME ZONE 'UTC' AS device_time,
            CASE 
                WHEN LOWER(value) IN ('1', 'true', 'on', 'yes') THEN TRUE
                ELSE FALSE
            END AS ignition
        FROM raw_telematics_data.states
        WHERE device_id = :device_id
        AND LOWER(state_name) IN ('ignition', 'ign', 'acc')
        AND device_time >= :start_time
        AND device_time < :end_time
        ORDER BY device_time
        """
        
        with self.engine.connect() as conn:
            df = pd.read_sql(
                text(query),
                conn,
                params={
                    "device_id": device_id,
                    "start_time": start_time,
                    "end_time": end_time
                }
            )
        
        return df

    def get_complete_fuel_data(
        self,
        object_info: ObjectInfo,
        start_time: datetime,
        end_time: datetime
    ) -> Tuple[pd.DataFrame, List[SensorInfo]]:
        """
        Get complete fuel data including sensor readings, GPS, and ignition.
        
        Returns:
            - Combined DataFrame with all data
            - List of sensor configurations
        """
        # Get sensors
        sensors = self.get_fuel_sensors_for_device(object_info.device_id)
        if not sensors:
            return pd.DataFrame(), []
        
        sensor_names = [s.input_label for s in sensors]
        
        # Get fuel data
        fuel_df = self.get_fuel_sensor_data(
            object_info.device_id,
            sensor_names,
            start_time,
            end_time
        )
        
        if fuel_df.empty:
            return pd.DataFrame(), sensors
        
        # Get GPS data
        gps_df = self.get_gps_data(
            object_info.device_id,
            start_time,
            end_time
        )
        
        # Get ignition data
        ign_df = self.get_ignition_states(
            object_info.device_id,
            start_time,
            end_time
        )
        
        # Merge fuel with GPS (nearest time match)
        fuel_df["device_time"] = pd.to_datetime(fuel_df["device_time"], utc=True)
        
        if not gps_df.empty:
            gps_df["device_time"] = pd.to_datetime(gps_df["device_time"], utc=True)
            fuel_df = pd.merge_asof(
                fuel_df.sort_values("device_time"),
                gps_df[["device_time", "lat", "lng", "speed_kmh", "hdop"]].sort_values("device_time"),
                on="device_time",
                direction="nearest",
                tolerance=pd.Timedelta("5min")
            )
        else:
            fuel_df["lat"] = None
            fuel_df["lng"] = None
            fuel_df["speed_kmh"] = None
            fuel_df["hdop"] = None
        
        # Merge with ignition
        if not ign_df.empty:
            ign_df["device_time"] = pd.to_datetime(ign_df["device_time"], utc=True)
            fuel_df = pd.merge_asof(
                fuel_df.sort_values("device_time"),
                ign_df[["device_time", "ignition"]].sort_values("device_time"),
                on="device_time",
                direction="backward",
                tolerance=pd.Timedelta("10min")
            )
        else:
            fuel_df["ignition"] = None
        
        # Add object info
        fuel_df["object_id"] = object_info.object_id
        fuel_df["object_label"] = object_info.object_label
        fuel_df["vehicle_id"] = object_info.vehicle_id
        fuel_df["vehicle_label"] = object_info.vehicle_label
        fuel_df["fuel_tank_volume"] = object_info.fuel_tank_volume
        
        return fuel_df.sort_values("device_time").reset_index(drop=True), sensors

    def close(self):
        """Close database connection"""
        if self._engine:
            self._engine.dispose()
            self._engine = None


def find_data_gaps(
    df: pd.DataFrame,
    time_column: str = "device_time",
    max_gap_minutes: int = 30
) -> List[Tuple[datetime, datetime]]:
    """
    Find gaps in time series data where no data was transmitted.
    
    Args:
        df: DataFrame with time column
        time_column: Name of datetime column
        max_gap_minutes: Maximum acceptable gap in minutes
    
    Returns:
        List of (start, end) tuples for each gap
    """
    if df.empty or len(df) < 2:
        return []
    
    times = pd.to_datetime(df[time_column]).sort_values()
    gaps = []
    
    for i in range(1, len(times)):
        gap = (times.iloc[i] - times.iloc[i-1]).total_seconds() / 60
        if gap > max_gap_minutes:
            gaps.append((times.iloc[i-1], times.iloc[i]))
    
    return gaps


