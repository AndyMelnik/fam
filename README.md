# FAM — Fuel Analytics Module

> Fuel sensor data processing module with Silver Layer support and Streamlit GUI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io)

## 🎯 Purpose

FAM (Fuel Analytics Module) is a module for processing telematics fuel sensor data that:

- **Reads raw data** from the Bronze layer (PostgreSQL)
- **Calibrates** sensor values to liters using calibration tables
- **Handles zero values** with intelligent interpolation
- **Applies hybrid smoothing** (Hampel → Median → EMA)
- **Detects events** (refuels and drains) with plateau-based volume calculation
- **Forms the Silver layer** for further analytics

## 📁 Project Structure

```
fam/
├── app.py                           # Streamlit GUI
├── config/
│   ├── fuel_sensor_data.etl.json   # ETL config for silver_data_layer.fuel_sensor_data
│   └── fuel_events.etl.json        # ETL config for silver_data_layer.fuel_events
├── src/
│   ├── __init__.py
│   ├── config.py             # Pydantic configuration models
│   ├── database.py           # DB connection and data extraction
│   ├── calibration.py        # Calibration and conversion to liters
│   ├── zero_handling.py      # Zero value handling
│   ├── smoothing.py          # Smoothing algorithms (Hampel, Median, EMA)
│   ├── detection.py          # Event detection with plateau refinement
│   ├── fuel_sensor_data.py   # Algorithm 1: Raw → Liters → Smoothed
│   └── fuel_events.py        # Algorithm 2: Event detection
├── output/                   # Output CSV files
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch GUI

```bash
streamlit run app.py
```

### 3. Usage

1. Enter PostgreSQL connection string
2. Select an object with fuel sensor from the list
3. Specify the time range
4. Configure processing parameters
5. Click **Process Data**

---

## 📊 Processing Algorithms

FAM implements two main algorithms that run sequentially:

### Algorithm 1: Fuel Sensor Data Processing

**Config:** `fuel_sensor_data.etl.json`  
**Output:** `silver_data_layer.fuel_sensor_data`

```
Raw Sensor Value → Calibration → Zero Handling → Smoothing → Liters
```

#### Step 1: Calibration (Raw → Liters)

Converts raw sensor values to physical liters using calibration tables stored in `sensor_description.calibration_data`.

- **Interpolation**: Linear interpolation between calibration points
- **Extrapolation**: Linear extrapolation beyond calibration range
- **Fallback**: If no calibration data, raw value is used as-is

```json
// Example calibration_data format
[
  {"in": 0, "out": 0},
  {"in": 100, "out": 25},
  {"in": 200, "out": 50},
  {"in": 400, "out": 100}
]
```

#### Step 2: Zero Handling

Handles problematic zero values that can occur due to sensor issues or ignition state.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mode` | Zero handling mode (see below) | `interpolate` |
| `startup_grace_period_min` | Grace period after ignition ON (minutes) | 5 |
| `min_zero_run_min` | Minimum duration for valid zero plateau (minutes) | 10 |
| `max_zero_level_l` | Maximum value considered as "zero" (liters) | 2.0 |

**Zero Handling Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `interpolate` | Replace invalid zeros with interpolated values | **Recommended**. Best for most sensors. Removes false zeros while preserving valid empty tank readings. |
| `remove` | Remove invalid zero points entirely | Use when you want to exclude uncertain data points from analysis |
| `keep` | Keep all zeros as-is | Use only when you trust all zero readings from the sensor |
| `auto` | Automatically detect and handle zeros based on context | Analyzes ignition state and stop duration to decide |

**How Invalid Zeros are Detected:**

1. **Startup zeros**: Zeros during the grace period after ignition ON are invalid (sensor warming up)
2. **Moving zeros**: Zeros when vehicle is moving (speed > 0) are usually invalid
3. **Short zero runs**: Brief zero periods (< `min_zero_run_min`) are often sensor glitches
4. **Valid zeros**: Extended zero plateaus during stops are likely real empty tank readings

#### Step 3: Hybrid Smoothing

Three-stage smoothing pipeline that preserves fuel level steps while removing noise:

```
Raw → Hampel Filter → Rolling Median → EMA → Smoothed
```

| Parameter | Description | Range |
|-----------|-------------|-------|
| `smoothing_level` | Overall smoothing intensity | 1-10 |

**Smoothing Level Effects:**

| Level | Hampel Window | Median Window | EMA Alpha | Description |
|-------|---------------|---------------|-----------|-------------|
| 1 | 60 sec | 3 points | 0.50 | Minimal smoothing, preserves all details |
| 5 | 480 sec | 9 points | 0.28 | Balanced smoothing |
| 10 | 900 sec | 15 points | 0.05 | Maximum smoothing, very smooth curve |

**Smoothing Stages:**

1. **Hampel Filter** (Outlier Removal)
   - Detects statistical outliers using MAD (Median Absolute Deviation)
   - Replaces outliers with local median
   - Parameters: `window_sec` (60-900), `n_sigma` (2.0-4.0)

2. **Rolling Median** (Local Smoothing)
   - Removes local noise while preserving edges
   - Parameters: `window_points` (3-15)

3. **Exponential Moving Average** (Global Smoothing)
   - Smooths the overall trend
   - Parameters: `alpha` (0.05-0.50)

---

### Algorithm 2: Fuel Event Detection

**Config:** `fuel_events.etl.json`  
**Output:** `silver_data_layer.fuel_events`

```
Smoothed Data → Candidate Detection → Clustering → Plateau Refinement → Validation → Events
```

#### Detection Thresholds

Events are detected when fuel level changes exceed configured thresholds:

**Refuel Thresholds** (fuel increases):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_step_l` | Minimum step size in liters | 3.0 |
| `min_step_pct_tank` | Minimum step as % of tank volume | 1.0% |
| `min_step_pct_level` | Minimum step as % of current level | 5.0% |

**Drain Thresholds** (fuel decreases):

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_step_l` | Minimum step size in liters | 3.0 |
| `min_step_pct_tank` | Minimum step as % of tank volume | 3.0% |
| `min_step_pct_level` | Minimum step as % of current level | 5.0% |

**Detection Mode:**

| Mode | Description |
|------|-------------|
| `absolute` | Use only absolute thresholds (`min_step_l`) |
| `relative` | Use only relative thresholds (`min_step_pct_*`) |
| `absolute_and_relative` | Event must exceed BOTH absolute AND relative thresholds |
| `absolute_or_relative` | Event must exceed EITHER absolute OR relative threshold |

#### Event Clustering

Groups nearby candidate points into single events:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `merge_window_min_min` | Minimum merge window (minutes) | 5 |
| `merge_window_min_max` | Maximum merge window (minutes) | 30 |
| `min_points_in_cluster` | Minimum points to form event | 1 |

The actual merge window is interpolated based on `smoothing_level`.

#### Plateau Detection (Accurate Volume Calculation)

**Key feature for accurate event volume calculation.**

Instead of summing deltas during the transition, FAM identifies stable fuel levels (plateaus) before and after each event:

```
[===Plateau Before===]  [Transition]  [===Plateau After===]
        ↓                                      ↓
   Start Level                            End Level
        
   Volume = |End Level - Start Level|
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `plateau_window_multiplier` | Multiplier for plateau search window | 2.0 |
| `plateau_stability_threshold_l` | Max fuel deviation within plateau (L) | 2.0 |

**How Plateau Detection Works:**

1. For each detected event, search backward for a stable plateau (before event)
2. Search forward for a stable plateau (after event)
3. A plateau is valid when fuel level varies by less than `stability_threshold_l` within the search window
4. Event volume = `|end_plateau_level - start_plateau_level|`

**Benefits:**
- More accurate volume calculation (not affected by transition noise)
- Better event boundary detection
- Handles gradual refueling/draining correctly

#### Event Filtering

Final filters to remove false positives:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_volume_l` | Minimum event volume (liters) | 3.0 |
| `min_volume_pct_tank` | Minimum event as % of tank | 2.0% |
| `allow_mixed_sign_cluster` | Allow clusters with both + and - deltas | false |

#### Context Validation

Additional validation based on vehicle context:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `max_valid_speed_kmh` | Max speed for valid event (km/h) | 5.0 |
| `max_hdop` | Max GPS HDOP for valid location | 5.0 |
| `use_ignition` | Validate against ignition state | true |
| `min_stop_duration_min` | Min stop duration for event (minutes) | 2 |

---

## ⚙️ Configuration Files

### Configuration Architecture

FAM uses **separate ETL configs** for each Silver layer target table:

| Config File | Target Table | Purpose |
|------------|--------------|---------|
| `fuel_sensor_data.etl.json` | `silver_data_layer.fuel_sensor_data` | Calibration, zero handling, smoothing |
| `fuel_events.etl.json` | `silver_data_layer.fuel_events` | Refuel/drain detection |

### ETL Config Structure

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "Fuel Sensor Data Silver Layer",
    "model_name": "hampel_median_ema_delta_cluster_v1",
    "model_version": "1.1.0",
    "tags": ["fuel", "sensor_data", "silver_layer", "prod"]
  },
  "parameters": {
    "processing_window_hours": 48,
    "smoothing": {...},
    "zero_handling": {...},
    "detection": {...}
  },
  "target": {
    "schema": "silver_data_layer",
    "table": "fuel_sensor_data",
    "primary_keys": ["device_id", "ts_utc", "processing_version"],
    "write_mode": "upsert",
    "idempotent": true
  },
  "scheduler": {
    "cron_expression": "0 * * * *",
    "enabled": true
  },
  "security": {
    "allowed_roles": ["airflow", "data_platform"],
    "no_secrets_in_config": true
  }
}
```

### Config Security

**Prohibited** in JSON configs:
- Passwords and tokens
- Database connection strings
- API keys

All credentials are loaded from:
- Secret Manager / Vault
- Environment Variables
- Airflow Connections

---

## 💾 Silver Layer Schema

### fuel_sensor_data

| Column | Type | Description |
|--------|------|-------------|
| device_id | BIGINT | Device ID |
| object_id | BIGINT | Object ID |
| vehicle_id | BIGINT | Vehicle ID |
| ts_utc | TIMESTAMPTZ | Timestamp (UTC) |
| lat | FLOAT | Latitude |
| lng | FLOAT | Longitude |
| sensor_raw_value | FLOAT | Original raw sensor value |
| fuel_level_l_raw | FLOAT | After calibration (liters) |
| fuel_level_l | FLOAT | After smoothing (liters) |
| fuel_level_norm | FLOAT | Normalized (0-1) |
| calibration_used | BOOLEAN | Whether calibration was applied |
| calibration_points | INT | Number of calibration points |
| processing_version | TEXT | Processing algorithm version |

### fuel_events

| Column | Type | Description |
|--------|------|-------------|
| event_id | UUID | Unique event ID |
| device_id | BIGINT | Device ID |
| event_type | TEXT | 'refuel' or 'drain' |
| start_datetime | TIMESTAMPTZ | Event start time |
| end_datetime | TIMESTAMPTZ | Event end time |
| start_level_l | FLOAT | Fuel level before event (liters) |
| end_level_l | FLOAT | Fuel level after event (liters) |
| volume_change_l | FLOAT | Absolute volume change (liters) |
| signed_volume_l | FLOAT | Signed volume (+refuel/-drain) |
| samples_in_event | INT | Number of data points in event |
| confidence | FLOAT | Detection confidence (0-1) |
| start_lat | FLOAT | Start location latitude |
| start_lng | FLOAT | Start location longitude |

---

## 🔧 Programmatic Usage

### Using Separate Processors (Recommended)

```python
from src.fuel_sensor_data import FuelSensorDataProcessor
from src.fuel_events import FuelEventsProcessor
from src.config import load_fuel_sensor_data_config, load_fuel_events_config
from src.database import DatabaseConnector
from datetime import datetime, timedelta

# Load configs
sensor_config = load_fuel_sensor_data_config()
events_config = load_fuel_events_config()

# Connect to database
db = DatabaseConnector("postgresql://user:pass@host/db")
objects = db.get_objects_with_fuel_sensors()

# Select an object
obj = objects[0]
start_time = datetime.now() - timedelta(days=2)
end_time = datetime.now()

# Algorithm 1: Process sensor data
sensor_processor = FuelSensorDataProcessor(
    config=sensor_config,
    db_connector=db
)
sensor_processor.load_data_from_db(obj, start_time, end_time)
fuel_data, smoothing_stats = sensor_processor.process()

# Algorithm 2: Detect events
events_processor = FuelEventsProcessor(config=events_config)
events_processor.load_fuel_data(fuel_data, obj)
events = events_processor.detect_events()

# Get Silver layer DataFrames
fuel_df = sensor_processor.get_silver_layer_dataframe()
events_df = events_processor.get_silver_layer_dataframe()

# Get summaries
print(sensor_processor.get_summary())
print(events_processor.get_summary())
```

### Adjusting Parameters at Runtime

```python
# Modify smoothing level
sensor_config.parameters.smoothing.smoothing_level = 7

# Modify detection thresholds
events_config.parameters.detection.refuel.min_step_l = 5.0
events_config.parameters.detection.drain.min_step_l = 5.0

# Modify plateau detection
events_config.parameters.detection.cluster.plateau_window_multiplier = 2.5
events_config.parameters.detection.cluster.plateau_stability_threshold_l = 1.5
```

---

## 📝 Airflow Integration

```python
from src.fuel_sensor_data import FuelSensorDataProcessor
from src.fuel_events import FuelEventsProcessor
from src.config import load_fuel_sensor_data_config, load_fuel_events_config
from src.database import DatabaseConnector
from airflow.models import Variable
from airflow.decorators import task

sensor_config = load_fuel_sensor_data_config()
events_config = load_fuel_events_config()

@task
def process_fuel_data(object_id: int, **context):
    db = DatabaseConnector(Variable.get("TELEMATICS_DB_URL"))
    obj = db.get_object_by_id(object_id)
    
    # Algorithm 1
    sensor_proc = FuelSensorDataProcessor(sensor_config, db)
    sensor_proc.load_data_from_db(
        obj,
        context["data_interval_start"],
        context["data_interval_end"]
    )
    fuel_data, _ = sensor_proc.process()
    
    # Algorithm 2
    events_proc = FuelEventsProcessor(events_config)
    events_proc.load_fuel_data(fuel_data, obj)
    events_proc.detect_events()
    
    return {
        "fuel_records": len(fuel_data),
        "events_detected": len(events_proc.events)
    }
```

### Scheduler Settings in Config

```json
"scheduler": {
  "cron_expression": "0 * * * *",
  "enabled": true,
  "max_runtime_sec": 1500,
  "max_concurrent_runs": 1,
  "backfill_allowed": false
}
```

---

## 🎨 GUI Features

- 🔌 PostgreSQL database connection
- 📋 Object selection with sensor diagnostics (shows data availability)
- 🔬 Calibration visualization (Raw → Liters conversion)
- ⚙️ Real-time parameter configuration
- 📊 Interactive visualization with Plotly
- 📄 ETL config viewing and editing
- 📥 Export to CSV

---

## 📄 License

MIT License
