# FAM — Fuel Analytics Module

> Fuel sensor data processing module with Silver Layer support and Streamlit GUI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io)

## 🎯 Purpose

FAM (Fuel Analytics Module) is a module for processing telematics fuel sensor data that:

- **Reads raw data** from the Bronze layer (PostgreSQL)
- **Calibrates** sensor values to liters
- **Applies hybrid smoothing** (Hampel → Median → EMA)
- **Detects events** (refuels and drains) with clustering
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
│   ├── smoothing.py          # Smoothing algorithms
│   ├── detection.py          # Event detection
│   ├── fuel_sensor_data.py   # Silver layer processor (data)
│   └── fuel_events.py        # Silver layer processor (events)
├── output/                   # Output CSV files
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch GUI

```bash
streamlit run app.py
```

### 3. Usage

1. Enter PostgreSQL connection string
2. Select an object from the list
3. Specify the time range
4. Configure processing parameters
5. Click **Process Data**

## ⚙️ Configuration

### Configuration Architecture

FAM uses **separate ETL configs** for each Silver layer target table:

| Config File | Target Table | Purpose |
|------------|--------------|---------|
| `fuel_sensor_data.etl.json` | `silver_data_layer.fuel_sensor_data` | Smoothing, zero handling |
| `fuel_events.etl.json` | `silver_data_layer.fuel_events` | Refuel/drain detection |

### ETL Config Structure

Each config contains:

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
    "zero_handling": {...}
  },
  "sql_template": {
    "engine": "postgres",
    "query": " ",
    "parameters": {...}
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

The following are **prohibited** in JSON configs:
- Passwords and tokens
- Database connection strings
- API keys

All credentials are loaded from:
- Secret Manager / Vault
- Environment Variables
- Airflow Connections

### Main Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `smoothing_level` | Smoothing intensity | 0.0 - 1.0 |
| `min_step_l` | Minimum step for detection (L) | 0.1 - 100.0 |
| `min_volume_l` | Minimum event volume (L) | 0.1 - 500.0 |

### Smoothing Modes

- **full** — complete pipeline (Hampel → Median → EMA)
- **hampel_only** — outlier removal only
- **median_only** — rolling median only
- **ema_only** — exponential smoothing only

## 📊 Processing Algorithm

### 1. Data Extraction

```sql
SELECT i.device_id, i.device_time, i.value, ...
FROM raw_telematics_data.inputs i
JOIN raw_business_data.sensor_description sd ON ...
WHERE sd.sensor_type = 'fuel'
```

### 2. Calibration

- Apply `less` / `more` thresholds
- Apply `multiplier` / `divider`
- Linear interpolation using `calibration_data` table

### 3. Zero Handling

- Grace period after ignition ON
- Valid zero plateaus at stops
- Interpolation of invalid zeros

### 4. Hybrid Smoothing

```
raw → Hampel (outliers) → Median (local) → EMA (global)
```

### 5. Event Detection

- Delta-based detector with adaptive thresholds
- Clustering of nearby points
- Context validation (speed, ignition)

## 💾 Silver Layer

### fuel_sensor_data

| Column | Type | Description |
|--------|------|-------------|
| device_id | BIGINT | Device ID |
| ts_utc | TIMESTAMPTZ | Timestamp |
| fuel_level_l_raw | FLOAT | Raw value (L) |
| fuel_level_l | FLOAT | Smoothed value (L) |
| fuel_level_norm | FLOAT | Normalized (0-1) |

### fuel_events

| Column | Type | Description |
|--------|------|-------------|
| event_type | TEXT | 'refuel' or 'drain' |
| start_datetime | TIMESTAMPTZ | Event start |
| end_datetime | TIMESTAMPTZ | Event end |
| volume_change_l | FLOAT | Volume change (L) |
| signed_volume_l | FLOAT | With sign (+/-) |

## 🔧 Programmatic Usage

### With New ETL Configs (Recommended)

```python
from src.fuel_events import FuelProcessingPipeline
from src.config import load_fuel_sensor_data_config, load_fuel_events_config
from src.database import DatabaseConnector
from datetime import datetime, timedelta

# Load configs
sensor_config = load_fuel_sensor_data_config()
events_config = load_fuel_events_config()

# Connect
db = DatabaseConnector("postgresql://user:pass@host/db")
objects = db.get_objects_with_fuel_sensors()

# Process with ETL configs
pipeline = FuelProcessingPipeline(
    sensor_config=sensor_config,
    events_config=events_config,
    db_connector=db
)

fuel_df, events_df, summary = pipeline.run(
    objects[0],
    datetime.now() - timedelta(days=2),
    datetime.now()
)

# Config info
print(pipeline.get_configs_info())

# Save results
pipeline.save_results("output/")
```

### Legacy Mode (Backwards Compatible)

```python
from src.fuel_events import FuelProcessingPipeline
from src.config import load_default_config
from src.database import DatabaseConnector

# Load merged config (from ETL configs)
config = load_default_config()

# Process
pipeline = FuelProcessingPipeline(config=config, db_connector=db)
fuel_df, events_df, summary = pipeline.run(...)
```

## 📝 Airflow Integration

Modules support Airflow DAG integration:

```python
from src.fuel_events import process_fuel_events
from src.config import load_fuel_sensor_data_config, load_fuel_events_config
from airflow.models import Variable

# Load configs from files
sensor_config = load_fuel_sensor_data_config()
events_config = load_fuel_events_config()

@task
def process_fuel_data(object_id: int, **context):
    fuel_df, events_df, summary = process_fuel_events(
        connection_string=Variable.get("TELEMATICS_DB_URL"),
        object_id=object_id,
        start_time=context["data_interval_start"],
        end_time=context["data_interval_end"],
        sensor_config=sensor_config,
        events_config=events_config
    )
    return summary
```

### Scheduler in Config

ETL configs contain Airflow settings:

```json
"scheduler": {
  "cron_expression": "0 * * * *",
  "enabled": true,
  "max_runtime_sec": 1500,
  "max_concurrent_runs": 1,
  "backfill_allowed": false
}
```

## 🎨 GUI Features

- 🔌 PostgreSQL database connection
- 📋 Object selection from list
- ⚙️ Real-time parameter configuration
- 📊 Interactive visualization with Plotly
- 📄 ETL config viewing and editing
- 📥 Export to CSV

## 📄 License

MIT License
