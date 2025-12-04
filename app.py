"""
Streamlit GUI for Fuel Analytics Module
Debug and testing interface for fuel sensor data processing

Orchestrates two algorithms:
1. FuelSensorDataProcessor - smoothing and conversion to liters
2. FuelEventsProcessor - refuel/drain event detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import FAM modules
from src.config import (
    FuelSensorDataETLConfig, 
    FuelEventsETLConfig,
    load_fuel_sensor_data_config,
    load_fuel_events_config
)
from src.database import DatabaseConnector, ObjectInfo
from src.fuel_sensor_data import FuelSensorDataProcessor
from src.fuel_events import FuelEventsProcessor

# Page config
st.set_page_config(
    page_title="FAM - Fuel Analytics Module",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'JetBrains Mono', monospace;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: #e94560;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border-left: 4px solid #e94560;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #e94560;
        color: #fff;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #e94560;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .config-section {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .config-badge {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.2rem;
    }
    
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    }
    
    .stButton button {
        background: linear-gradient(135deg, #e94560 0%, #c73659 100%);
        color: white;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #ff6b6b 0%, #e94560 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(233, 69, 96, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state variables"""
    # Load configs
    sensor_config = load_fuel_sensor_data_config()
    events_config = load_fuel_events_config()
    
    defaults = {
        "db_connected": False,
        "db_connector": None,
        "objects_list": [],
        "objects_details_df": None,
        "selected_object": None,
        "sensor_config": sensor_config,
        "events_config": events_config,
        "processed_fuel_df": None,
        "events_df": None,
        "events": [],
        "data_gaps": [],
        "summary": None,
        "sensor_processor": None  # For calibration visualization
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Processing Pipeline
# =============================================================================

def run_processing_pipeline(
    db_connector: DatabaseConnector,
    object_info: ObjectInfo,
    start_time: datetime,
    end_time: datetime,
    sensor_config: FuelSensorDataETLConfig,
    events_config: FuelEventsETLConfig
) -> tuple:
    """
    Run the complete fuel processing pipeline.
    
    Algorithm 1: FuelSensorDataProcessor (smoothing, conversion to liters)
    Algorithm 2: FuelEventsProcessor (event detection)
    
    Returns:
        Tuple of (fuel_df, events_df, summary, sensor_processor, events_processor)
    """
    # Algorithm 1: Sensor Data Processing
    sensor_processor = FuelSensorDataProcessor(
        config=sensor_config,
        db_connector=db_connector
    )
    
    # Load raw data
    raw_data = sensor_processor.load_data_from_db(object_info, start_time, end_time)
    
    if raw_data is None or raw_data.empty:
        sensors = sensor_processor.sensors
        if not sensors:
            raise ValueError(
                f"No fuel sensors configured for device {object_info.device_id}. "
                f"Check raw_business_data.sensor_description table."
            )
        else:
            sensor_names = [s.input_label for s in sensors]
            # Get available sensor names for diagnostics
            available_sensors = db_connector.get_available_sensor_names(
                object_info.device_id, start_time, end_time
            )
            
            error_msg = (
                f"No fuel data found for device {object_info.device_id} "
                f"in period {start_time} to {end_time}.\n\n"
                f"**Expected sensors (from sensor_description):** {sensor_names}\n\n"
            )
            
            if available_sensors:
                error_msg += f"**Available sensors (in inputs table):** {available_sensors[:20]}"
                if len(available_sensors) > 20:
                    error_msg += f" ... and {len(available_sensors) - 20} more"
                error_msg += "\n\n⚠️ Check if input_label matches sensor_name in inputs table!"
            else:
                error_msg += "**No sensor data at all found in inputs table for this device and period.**"
            
            raise ValueError(error_msg)
    
    # Process: Calibration → Zero Handling → Smoothing
    fuel_data, smoothing_stats = sensor_processor.process()
    
    # Algorithm 2: Event Detection
    events_processor = FuelEventsProcessor(config=events_config)
    events_processor.load_fuel_data(fuel_data, object_info)
    events = events_processor.detect_events()
    
    # Get output DataFrames
    fuel_df = sensor_processor.get_silver_layer_dataframe()
    events_df = events_processor.get_silver_layer_dataframe()
    
    # Combined summary
    summary = {
        "sensor_data": sensor_processor.get_summary(),
        "events": events_processor.get_summary()
    }
    
    return fuel_df, events_df, summary, sensor_processor, events_processor


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    """Render main header"""
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2rem;">⛽ FAM — Fuel Analytics Module</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #aaa;">
            Bronze → Silver Layer Processing | Debug & Testing Interface
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_connection_sidebar():
    """Render database connection sidebar"""
    st.sidebar.markdown("## 🔌 Database Connection")
    
    conn_string = st.sidebar.text_input(
        "PostgreSQL URL",
        value="postgresql://user:password@localhost:5432/telematics",
        type="password",
        help="Format: postgresql://user:password@host:port/database"
    )
    
    if st.sidebar.button("🔗 Connect to Database", use_container_width=True):
        try:
            with st.sidebar.status("Connecting...", expanded=True) as status:
                st.write("Testing connection...")
                db = DatabaseConnector(conn_string)
                
                if db.test_connection():
                    st.write("✅ Connection OK")
                    st.write("Loading objects with fuel sensors...")
                    
                    objects = db.get_objects_with_fuel_sensors()
                    objects_details_df = db.get_objects_with_sensors_details()
                    
                    if objects:
                        st.session_state.db_connector = db
                        st.session_state.db_connected = True
                        st.session_state.objects_list = objects
                        st.session_state.objects_details_df = objects_details_df
                        status.update(label=f"✅ Connected! Found {len(objects)} objects", state="complete")
                    else:
                        st.warning("⚠️ No objects with fuel sensors found")
                        status.update(label="⚠️ No fuel sensors found", state="error")
                else:
                    status.update(label="❌ Connection failed", state="error")
        except Exception as e:
            st.sidebar.error(f"❌ Error: {str(e)}")
    
    # Object and time selection
    if st.session_state.db_connected:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Select Object")
        
        if st.session_state.objects_list:
            object_options = {
                f"{obj.object_label} (ID: {obj.object_id}, Device: {obj.device_id})": obj
                for obj in st.session_state.objects_list
            }
            
            selected_label = st.sidebar.selectbox("Object", options=list(object_options.keys()))
            st.session_state.selected_object = object_options[selected_label]
            
            obj = st.session_state.selected_object
            st.sidebar.markdown(f"""
            <div class="config-section">
                <small>
                    <b>Vehicle:</b> {obj.vehicle_label or 'N/A'}<br>
                    <b>Tank:</b> {obj.fuel_tank_volume or 'Unknown'} L
                </small>
            </div>
            """, unsafe_allow_html=True)
        
        st.sidebar.markdown("### 📅 Time Range")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=2))
        with col2:
            start_time = st.time_input("Start Time", value=datetime.strptime("00:00", "%H:%M").time())
        
        col3, col4 = st.sidebar.columns(2)
        with col3:
            end_date = st.date_input("End Date", value=datetime.now())
        with col4:
            end_time = st.time_input("End Time", value=datetime.strptime("23:59", "%H:%M").time())
        
        st.session_state.start_datetime = datetime.combine(start_date, start_time)
        st.session_state.end_datetime = datetime.combine(end_date, end_time)


def render_config_info():
    """Render config info badges"""
    sensor_config = st.session_state.sensor_config
    events_config = st.session_state.events_config
    
    if sensor_config and events_config:
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <span class="config-badge">📊 {sensor_config.metadata.model_name}</span>
            <span class="config-badge">v{sensor_config.metadata.model_version}</span>
            <span class="config-badge">🎯 {sensor_config.target.schema_name}.{sensor_config.target.table}</span>
            <span class="config-badge">⚡ {events_config.target.table}</span>
        </div>
        """, unsafe_allow_html=True)


def render_objects_table():
    """Render table with objects and their fuel sensor configurations"""
    if st.session_state.objects_details_df is None or st.session_state.objects_details_df.empty:
        st.info("No objects with fuel sensors found")
        return
    
    df = st.session_state.objects_details_df.copy()
    
    # Rename columns for display (9 columns: object_id, object_label, device_id, sensor_label, input_label, sensor_type, sensor_units, calibration_data, has_data)
    num_cols = len(df.columns)
    if num_cols == 9:
        df.columns = ["Object ID", "Object Label", "Device ID", "Sensor Label", "Input Label", "Sensor Type", "Units", "Calibration", "Has Data"]
    elif num_cols == 8:
        # Could be with or without has_data
        if "has_data" in df.columns:
            df.columns = ["Object ID", "Object Label", "Device ID", "Sensor Label", "Input Label", "Sensor Type", "Units", "Has Data"]
        else:
            df.columns = ["Object ID", "Object Label", "Device ID", "Sensor Label", "Input Label", "Sensor Type", "Units", "Calibration"]
    else:
        df.columns = ["Object ID", "Object Label", "Device ID", "Sensor Label", "Input Label", "Sensor Type", "Calibration"]
    
    st.markdown("### 📋 Objects with Fuel Sensors")
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Object ID": st.column_config.NumberColumn("Object ID", width="small"),
            "Device ID": st.column_config.NumberColumn("Device ID", width="small"),
            "Units": st.column_config.TextColumn("Units", width="small"),
            "Calibration": st.column_config.TextColumn("Calibration", width="medium"),
            "Has Data": st.column_config.TextColumn("Has Data (7d)", width="small"),
        }
    )
    st.caption(f"Total: {len(df)} sensor configurations across {df['Object ID'].nunique()} objects")


def render_config_panel():
    """Render configuration panel"""
    sensor_config = st.session_state.sensor_config
    events_config = st.session_state.events_config
    
    tabs = st.tabs([
        "🎚️ Algorithm 1: Smoothing", 
        "🔍 Algorithm 2: Detection",
        "📊 Sensor Config JSON",
        "⚡ Events Config JSON"
    ])
    
    # Algorithm 1: Smoothing config
    with tabs[0]:
        st.markdown("#### Fuel Sensor Data Processing")
        st.caption("Config: `fuel_sensor_data.etl.json`")
        st.caption("Pipeline: **Calibration** → **Zero Handling** → **Smoothing** (Hampel → Median → EMA)")
        
        smoothing_level = st.slider(
            "Smoothing Intensity",
            min_value=1,
            max_value=10,
            value=int(sensor_config.parameters.smoothing.smoothing_level),
            step=1,
            help="1 = minimal smoothing, 10 = maximum smoothing"
        )
        sensor_config.parameters.smoothing.smoothing_level = smoothing_level
        
        # Show calculated parameters
        level = (smoothing_level - 1) / 9.0
        hampel_cfg = sensor_config.parameters.smoothing.hampel
        median_cfg = sensor_config.parameters.smoothing.median
        ema_cfg = sensor_config.parameters.smoothing.ema
        
        def lerp(min_v, max_v, t): return min_v + (max_v - min_v) * t
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="config-section">
                <small>
                    <b>Calculated Parameters:</b><br>
                    Hampel Window: {lerp(hampel_cfg.window_min_sec, hampel_cfg.window_max_sec, level):.0f}s<br>
                    Hampel σ: {lerp(hampel_cfg.sigma_max, hampel_cfg.sigma_min, level):.2f}
                </small>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="config-section">
                <small>
                    <b>&nbsp;</b><br>
                    Median Window: {int(lerp(median_cfg.window_min_points, median_cfg.window_max_points, level))} pts<br>
                    EMA α: {lerp(ema_cfg.alpha_max, ema_cfg.alpha_min, level):.3f}
                </small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("**Zero Handling**")
        col3, col4 = st.columns(2)
        with col3:
            sensor_config.parameters.zero_handling.mode = st.selectbox(
                "Mode",
                options=["auto", "keep", "interpolate", "drop"],
                index=["auto", "keep", "interpolate", "drop"].index(sensor_config.parameters.zero_handling.mode)
            )
        with col4:
            sensor_config.parameters.zero_handling.max_zero_level_l = st.number_input(
                "Max Zero Level (L)",
                value=sensor_config.parameters.zero_handling.max_zero_level_l,
                min_value=0.0, max_value=10.0, step=0.5
            )
    
    # Algorithm 2: Detection config
    with tabs[1]:
        st.markdown("#### Fuel Events Detection")
        st.caption("Config: `fuel_events.etl.json`")
        st.caption("Pipeline: **Candidate Detection** → **Clustering** → **Context Validation** → **Filtering**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Refuel Thresholds**")
            events_config.parameters.detection.refuel.min_step_l = st.number_input(
                "Min Step (L)", value=events_config.parameters.detection.refuel.min_step_l,
                min_value=0.1, max_value=100.0, step=1.0, key="refuel_step"
            )
            events_config.parameters.detection.refuel.min_step_pct_tank = st.number_input(
                "Min % of Tank", value=events_config.parameters.detection.refuel.min_step_pct_tank,
                min_value=0.1, max_value=50.0, step=0.5, key="refuel_pct"
            )
        
        with col2:
            st.markdown("**Drain Thresholds**")
            events_config.parameters.detection.drain.min_step_l = st.number_input(
                "Min Step (L)", value=events_config.parameters.detection.drain.min_step_l,
                min_value=0.1, max_value=100.0, step=1.0, key="drain_step"
            )
            events_config.parameters.detection.drain.min_step_pct_tank = st.number_input(
                "Min % of Tank", value=events_config.parameters.detection.drain.min_step_pct_tank,
                min_value=0.1, max_value=50.0, step=0.5, key="drain_pct"
            )
        
        st.markdown("**Event Filtering**")
        col3, col4 = st.columns(2)
        with col3:
            events_config.parameters.detection.event.min_volume_l = st.number_input(
                "Min Event Volume (L)", value=events_config.parameters.detection.event.min_volume_l,
                min_value=0.1, max_value=500.0, step=1.0
            )
        with col4:
            events_config.parameters.detection.event.min_volume_pct_tank = st.number_input(
                "Min % of Tank for Event", value=events_config.parameters.detection.event.min_volume_pct_tank,
                min_value=0.1, max_value=100.0, step=0.5
            )
        
        st.markdown("**Plateau Detection** *(for accurate volume calculation)*")
        col5, col6 = st.columns(2)
        with col5:
            events_config.parameters.detection.cluster.plateau_window_multiplier = st.number_input(
                "Plateau Window Multiplier",
                value=float(events_config.parameters.detection.cluster.plateau_window_multiplier),
                min_value=1.0, max_value=5.0, step=0.1,
                help="Multiplier for merge_window to search for stable plateaus"
            )
        with col6:
            events_config.parameters.detection.cluster.stability_threshold_l = st.number_input(
                "Stability Threshold (L)",
                value=float(events_config.parameters.detection.cluster.stability_threshold_l),
                min_value=0.1, max_value=10.0, step=0.1,
                help="Max std dev (L) for considering a region as stable plateau"
            )
    
    # JSON editors
    with tabs[2]:
        st.markdown("#### Sensor Data Config")
        st.caption("`config/fuel_sensor_data.etl.json`")
        
        json_str = json.dumps(sensor_config.to_dict(), indent=2, ensure_ascii=False)
        edited_json = st.text_area("Edit JSON", value=json_str, height=400, key="sensor_json")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Apply Changes", key="apply_sensor"):
                try:
                    st.session_state.sensor_config = FuelSensorDataETLConfig(**json.loads(edited_json))
                    st.success("✅ Config updated!")
                except Exception as e:
                    st.error(f"❌ Invalid JSON: {e}")
        with col2:
            if st.button("💾 Save to File", key="save_sensor"):
                try:
                    sensor_config.to_json_file("config/fuel_sensor_data.etl.json")
                    st.success("✅ Saved!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with tabs[3]:
        st.markdown("#### Events Config")
        st.caption("`config/fuel_events.etl.json`")
        
        json_str = json.dumps(events_config.to_dict(), indent=2, ensure_ascii=False)
        edited_json = st.text_area("Edit JSON", value=json_str, height=400, key="events_json")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Apply Changes", key="apply_events"):
                try:
                    st.session_state.events_config = FuelEventsETLConfig(**json.loads(edited_json))
                    st.success("✅ Config updated!")
                except Exception as e:
                    st.error(f"❌ Invalid JSON: {e}")
        with col2:
            if st.button("💾 Save to File", key="save_events"):
                try:
                    events_config.to_json_file("config/fuel_events.etl.json")
                    st.success("✅ Saved!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")


def render_calibration_visualization(fuel_df: pd.DataFrame, sensor_processor):
    """
    Render calibration visualization showing raw sensor values → calibrated output.
    Shows measurement units and clipping statistics.
    Only shown if calibration was used.
    """
    from plotly.subplots import make_subplots
    
    # Get measurement units from processor
    units = getattr(sensor_processor, 'measurement_units', 'L')
    units_desc = getattr(sensor_processor, 'measurement_units_description', 'Liters')
    
    # Check if calibration was used
    if not fuel_df["calibration_used"].any():
        # Get sensor units directly if no calibration
        sensors = sensor_processor.sensors if sensor_processor else []
        sensor_units = sensors[0].sensor_units if sensors else None
        units_msg = f" Values are displayed in **{units_desc}** ({units})." if units else ""
        st.info(f"ℹ️ No calibration data available for this sensor. Raw values are used directly.{units_msg}")
        return
    
    # Check if we have raw sensor values
    if "sensor_raw_value" not in fuel_df.columns or fuel_df["sensor_raw_value"].isna().all():
        st.warning("⚠️ Raw sensor values not available for visualization.")
        return
    
    # Get calibration table from sensor
    sensors = sensor_processor.sensors
    if not sensors:
        return
    
    sensor = sensors[0]
    calibration_data = sensor.calibration_data
    
    if not calibration_data or len(calibration_data) < 2:
        st.info("ℹ️ Calibration table not available or has insufficient points.")
        return
    
    # Display measurement units info
    st.markdown(f"**Measurement Units:** {units_desc} ({units})")
    
    # Create visualization with 2 subplots
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        subplot_titles=(
            f"📊 Raw Sensor Value vs Calibrated Output ({units})",
            f"📐 Calibration Curve"
        ),
        horizontal_spacing=0.1
    )
    
    COLORS = {
        "raw": "#f97316",      # Orange for raw
        "calibrated": "#2563eb",  # Blue for calibrated
        "curve": "#10b981",    # Green for calibration curve
        "points": "#ef4444",   # Red for calibration points
        "clipped": "#dc2626",  # Red for clipped regions
    }
    
    # Plot 1: Time series comparison
    df_valid = fuel_df.dropna(subset=["sensor_raw_value", "fuel_level_l_raw"])
    
    # Raw sensor values (left Y axis)
    fig.add_trace(
        go.Scatter(
            x=df_valid["ts_utc"],
            y=df_valid["sensor_raw_value"],
            mode="lines",
            name="Raw Sensor Value",
            line=dict(color=COLORS["raw"], width=2),
            opacity=0.8,
            yaxis="y1",
            hovertemplate="<b>Raw:</b> %{y:.1f}<br><b>Time:</b> %{x}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Calibrated output (right Y axis)
    fig.add_trace(
        go.Scatter(
            x=df_valid["ts_utc"],
            y=df_valid["fuel_level_l_raw"],
            mode="lines",
            name=f"Calibrated ({units})",
            line=dict(color=COLORS["calibrated"], width=2),
            opacity=0.8,
            yaxis="y2",
            hovertemplate=f"<b>{units}:</b> %{{y:.1f}}<br><b>Time:</b> %{{x}}<extra></extra>"
        ),
        row=1, col=1
    )
    
    # Plot 2: Calibration curve
    cal_points = sorted(calibration_data, key=lambda x: x["in"])
    cal_in = [p["in"] for p in cal_points]
    cal_out = [p["out"] for p in cal_points]
    
    # Calibration range boundaries
    cal_min_in = min(cal_in)
    cal_max_in = max(cal_in)
    cal_min_out = min(cal_out)
    cal_max_out = max(cal_out)
    
    # Interpolated curve
    import numpy as np
    if len(cal_in) > 1:
        x_interp = np.linspace(cal_min_in, cal_max_in, 100)
        y_interp = np.interp(x_interp, cal_in, cal_out)
        
        fig.add_trace(
            go.Scatter(
                x=x_interp,
                y=y_interp,
                mode="lines",
                name="Calibration Curve",
                line=dict(color=COLORS["curve"], width=3),
                hovertemplate=f"<b>Raw:</b> %{{x:.1f}}<br><b>{units}:</b> %{{y:.1f}}<extra></extra>"
            ),
            row=1, col=2
        )
    
    # Calibration points
    fig.add_trace(
        go.Scatter(
            x=cal_in,
            y=cal_out,
            mode="markers",
            name="Calibration Points",
            marker=dict(color=COLORS["points"], size=10, symbol="circle"),
            hovertemplate=f"<b>Raw:</b> %{{x:.1f}}<br><b>{units}:</b> %{{y:.1f}}<extra></extra>"
        ),
        row=1, col=2
    )
    
    # Add data range overlay on calibration curve
    raw_min = df_valid["sensor_raw_value"].min()
    raw_max = df_valid["sensor_raw_value"].max()
    
    # Highlight valid data range
    fig.add_vrect(
        x0=max(raw_min, cal_min_in), x1=min(raw_max, cal_max_in),
        fillcolor="rgba(37, 99, 235, 0.1)",
        layer="below",
        line_width=0,
        row=1, col=2
    )
    
    # Add calibration range boundaries as vertical lines
    fig.add_vline(x=cal_min_in, line_dash="dash", line_color=COLORS["clipped"], 
                  annotation_text=f"Min: {cal_min_in:.0f}", annotation_position="top left", row=1, col=2)
    fig.add_vline(x=cal_max_in, line_dash="dash", line_color=COLORS["clipped"], 
                  annotation_text=f"Max: {cal_max_in:.0f}", annotation_position="top right", row=1, col=2)
    
    fig.add_annotation(
        x=(max(raw_min, cal_min_in) + min(raw_max, cal_max_in)) / 2,
        y=cal_max_out * 0.95,
        text=f"Data range: {raw_min:.0f} - {raw_max:.0f}",
        showarrow=False,
        font=dict(size=10, color="#2563eb"),
        bgcolor="rgba(255,255,255,0.8)",
        row=1, col=2
    )
    
    # Layout
    fig.update_layout(
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.9)"
        ),
        plot_bgcolor="#f8fafc",
        paper_bgcolor="#ffffff",
        margin=dict(l=60, r=60, t=80, b=40)
    )
    
    # First subplot: dual Y-axis
    fig.update_yaxes(
        title=dict(text="Raw Sensor Value", font=dict(color=COLORS["raw"])),
        tickfont=dict(color=COLORS["raw"]),
        gridcolor="#e2e8f0",
        row=1, col=1
    )
    
    # Add secondary Y-axis for first subplot
    fig.update_layout(
        yaxis2=dict(
            title=dict(text=units, font=dict(color=COLORS["calibrated"])),
            tickfont=dict(color=COLORS["calibrated"]),
            anchor="x",
            overlaying="y",
            side="right"
        )
    )
    
    # Second subplot axes
    fig.update_xaxes(title_text="Raw Sensor Value", gridcolor="#e2e8f0", row=1, col=2)
    fig.update_yaxes(title_text=units, gridcolor="#e2e8f0", row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate clipping statistics
    clipped_low = fuel_df.get("calibration_clipped_low", pd.Series([False])).sum()
    clipped_high = fuel_df.get("calibration_clipped_high", pd.Series([False])).sum()
    total_clipped = clipped_low + clipped_high
    clipped_pct = (total_clipped / len(fuel_df) * 100) if len(fuel_df) > 0 else 0
    
    # Show calibration stats
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Calibration Points", len(cal_points))
    with col2:
        st.metric("Calibration Range (Raw)", f"{cal_min_in:.0f} - {cal_max_in:.0f}")
    with col3:
        st.metric(f"Output Range ({units})", f"{cal_min_out:.0f} - {cal_max_out:.0f}")
    with col4:
        st.metric("Clipped (Total)", f"{total_clipped} ({clipped_pct:.1f}%)")
    with col5:
        st.metric("Clipped Low / High", f"{int(clipped_low)} / {int(clipped_high)}")
    
    # Show warning if significant clipping
    if clipped_pct > 5:
        st.warning(f"⚠️ **{clipped_pct:.1f}%** of values were outside the calibration range and were clipped to boundary values. "
                   f"Consider extending the calibration table to cover the full sensor range ({raw_min:.0f} - {raw_max:.0f}).")


def render_visualization(fuel_df: pd.DataFrame, events: list, data_gaps: list, show_speed: bool = False, measurement_units: str = "L"):
    """Render main visualization with measurement units"""
    
    COLORS = {
        "raw_line": "#93c5fd",
        "smoothed_line": "#1d4ed8",
        "gap_bg": "rgba(229, 231, 235, 0.6)",
        "refuel_bg": "rgba(187, 247, 208, 0.5)",
        "refuel_border": "#16a34a",
        "drain_bg": "rgba(254, 202, 202, 0.5)",
        "drain_border": "#dc2626",
        "speed_line": "#ea580c",
        "plot_bg": "#f8fafc",
        "grid": "#e2e8f0",
    }
    
    units = measurement_units  # Use provided units
    
    if show_speed:
        fig = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True,
                          vertical_spacing=0.08, subplot_titles=(f"📊 Fuel Level ({units})", "🚗 Speed (km/h)"))
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=(f"📊 Fuel Level ({units})",))
    
    # Data gaps
    for gap_start, gap_end in data_gaps:
        fig.add_vrect(x0=gap_start, x1=gap_end, fillcolor=COLORS["gap_bg"], layer="below", line_width=0, row=1, col=1)
        if show_speed:
            fig.add_vrect(x0=gap_start, x1=gap_end, fillcolor=COLORS["gap_bg"], layer="below", line_width=0, row=2, col=1)
        fig.add_annotation(x=gap_start + (gap_end - gap_start) / 2, y=0.95, yref="y domain", text="📡 No Data",
                          showarrow=False, font=dict(color="#64748b", size=11), bgcolor="rgba(241,245,249,0.9)",
                          bordercolor="#cbd5e1", borderwidth=1, borderpad=4, row=1, col=1)
    
    # Events
    for event in events:
        fill_color = COLORS["refuel_bg"] if event.event_type == "refuel" else COLORS["drain_bg"]
        border_color = COLORS["refuel_border"] if event.event_type == "refuel" else COLORS["drain_border"]
        symbol = "⛽ ▲" if event.event_type == "refuel" else "🚨 ▼"
        
        fig.add_vrect(x0=event.start_datetime, x1=event.end_datetime, fillcolor=fill_color, layer="below",
                     line=dict(color=border_color, width=1, dash="dot"), row=1, col=1)
        if show_speed:
            fig.add_vrect(x0=event.start_datetime, x1=event.end_datetime, fillcolor=fill_color, layer="below", line_width=0, row=2, col=1)
        
        y_max = fuel_df["fuel_level_l"].max() if fuel_df["fuel_level_l"].notna().any() else 100
        fig.add_annotation(x=event.start_datetime + (event.end_datetime - event.start_datetime) / 2,
                          y=y_max * 0.95, text=f"{symbol} {event.volume_change_l:.1f} {units}", showarrow=True,
                          arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor=border_color,
                          font=dict(color="#ffffff", size=12), bgcolor=border_color, bordercolor=border_color,
                          borderwidth=1, borderpad=5, row=1, col=1)
    
    # Data lines
    if "fuel_level_l_raw" in fuel_df.columns and fuel_df["fuel_level_l_raw"].notna().any():
        fig.add_trace(go.Scatter(x=fuel_df["ts_utc"], y=fuel_df["fuel_level_l_raw"], mode="lines", name="Raw (Calibrated)",
                                line=dict(color=COLORS["raw_line"], width=2), opacity=0.7,
                                hovertemplate=f"<b>Calibrated:</b> %{{y:.1f}} {units}<br><b>Time:</b> %{{x}}<extra></extra>"), row=1, col=1)
    
    if fuel_df["fuel_level_l"].notna().any():
        fig.add_trace(go.Scatter(x=fuel_df["ts_utc"], y=fuel_df["fuel_level_l"], mode="lines", name="Smoothed",
                                line=dict(color=COLORS["smoothed_line"], width=3),
                                hovertemplate=f"<b>Smoothed:</b> %{{y:.1f}} {units}<br><b>Time:</b> %{{x}}<extra></extra>"), row=1, col=1)
    
    if show_speed and "speed_kmh" in fuel_df.columns and fuel_df["speed_kmh"].notna().any():
        fig.add_trace(go.Scatter(x=fuel_df["ts_utc"], y=fuel_df["speed_kmh"].fillna(0), mode="lines", name="Speed",
                                line=dict(color=COLORS["speed_line"], width=1.5), fill="tozeroy",
                                fillcolor="rgba(251, 146, 60, 0.3)",
                                hovertemplate="<b>Speed:</b> %{y:.1f} km/h<br><b>Time:</b> %{x}<extra></extra>"), row=2, col=1)
    
    fig.update_layout(height=600 if show_speed else 500, showlegend=True,
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1),
                     hovermode="x unified", plot_bgcolor=COLORS["plot_bg"], paper_bgcolor="#ffffff",
                     font=dict(color="#1e293b", size=12), margin=dict(l=60, r=20, t=80, b=40))
    
    fig.update_xaxes(gridcolor=COLORS["grid"], linecolor="#cbd5e1", tickfont=dict(color="#475569"), showgrid=True)
    fig.update_yaxes(gridcolor=COLORS["grid"], linecolor="#cbd5e1", tickfont=dict(color="#475569"),
                    title_text=units, showgrid=True, row=1, col=1)
    if show_speed:
        fig.update_yaxes(gridcolor=COLORS["grid"], title_text="km/h", showgrid=True, row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Legend
    st.markdown("""
    <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; padding: 10px 16px; 
                background: #f1f5f9; border: 1px solid #e2e8f0; border-radius: 8px; margin-top: 8px;">
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="width: 20px; height: 4px; background: #93c5fd; border-radius: 2px;"></span>
            <span style="color: #475569; font-size: 13px;">Raw Data</span>
        </span>
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="width: 20px; height: 4px; background: #1d4ed8; border-radius: 2px;"></span>
            <span style="color: #475569; font-size: 13px;">Smoothed</span>
        </span>
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="width: 18px; height: 18px; background: rgba(187,247,208,0.7); border: 2px solid #16a34a; border-radius: 3px;"></span>
            <span style="color: #16a34a; font-size: 13px;">⛽ Refuel</span>
        </span>
        <span style="display: flex; align-items: center; gap: 6px;">
            <span style="width: 18px; height: 18px; background: rgba(254,202,202,0.7); border: 2px solid #dc2626; border-radius: 3px;"></span>
            <span style="color: #dc2626; font-size: 13px;">🚨 Drain</span>
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_events_table(events: list, measurement_units: str = "L"):
    """Render events table with measurement units"""
    if not events:
        st.info("No events detected")
        return
    
    units = measurement_units
    
    events_data = []
    for i, event in enumerate(events):
        events_data.append({
            "#": i + 1,
            "Type": "🟢 Refuel" if event.event_type == "refuel" else "🔴 Drain",
            "Start": event.start_datetime.strftime("%Y-%m-%d %H:%M"),
            "End": event.end_datetime.strftime("%Y-%m-%d %H:%M"),
            f"Start ({units})": f"{event.start_level_l:.1f}",
            f"End ({units})": f"{event.end_level_l:.1f}",
            f"Volume ({units})": f"{event.volume_change_l:.1f}",
            f"Signed ({units})": f"{event.signed_volume_l:+.1f}",
            "Samples": event.samples_in_event,
            "Confidence": f"{event.confidence:.0%}",
            "Location": f"({event.start_lat:.4f}, {event.start_lng:.4f})" if event.start_lat else "N/A"
        })
    
    st.dataframe(pd.DataFrame(events_data), use_container_width=True, hide_index=True)
    
    # Show volume calculation explanation
    st.caption("💡 Volume is calculated as the difference between stable plateau levels (end_level - start_level)")


def render_silver_layer_tables(fuel_df: pd.DataFrame, events_df: pd.DataFrame):
    """Render Silver layer output tables"""
    sensor_config = st.session_state.sensor_config
    events_config = st.session_state.events_config
    
    tab1, tab2 = st.tabs([
        f"📊 {sensor_config.target.schema_name}.{sensor_config.target.table}",
        f"⚡ {events_config.target.schema_name}.{events_config.target.table}"
    ])
    
    with tab1:
        st.dataframe(fuel_df.head(100), use_container_width=True, hide_index=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download CSV", fuel_df.to_csv(index=False), "fuel_sensor_data.csv", "text/csv")
        with col2:
            st.metric("Total Records", len(fuel_df))
    
    with tab2:
        if not events_df.empty:
            st.dataframe(events_df, use_container_width=True, hide_index=True)
            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 Download CSV", events_df.to_csv(index=False), "fuel_events.csv", "text/csv")
            with col2:
                st.metric("Total Events", len(events_df))
        else:
            st.info("No events detected")


def render_metrics(summary: dict, measurement_units: str = "L"):
    """Render processing metrics with measurement units"""
    sensor_data = summary.get("sensor_data", {})
    events_data = summary.get("events", {})
    units = measurement_units
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sensor_data.get('total_points', 0):,}</div>
            <div class="metric-label">Data Points</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{events_data.get('refuels', 0)}</div>
            <div class="metric-label">Refuels</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{events_data.get('drains', 0)}</div>
            <div class="metric-label">Drains</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{events_data.get('total_refuel_volume', 0):.0f} {units}</div>
            <div class="metric-label">Total Refueled</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{sensor_data.get('smoothing_stats', {}).get('outliers_removed', 0)}</div>
            <div class="metric-label">Outliers Removed</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application"""
    init_session_state()
    
    render_header()
    render_connection_sidebar()
    
    if not st.session_state.db_connected:
        st.info("👈 Connect to database to start")
        st.markdown("---")
        st.markdown("### 📋 Loaded ETL Configurations")
        render_config_info()
        with st.expander("⚙️ View/Edit Configurations", expanded=False):
            render_config_panel()
        return
    
    render_config_info()
    
    # Objects table with sensor configurations
    with st.expander("📋 Objects with Fuel Sensors", expanded=True):
        render_objects_table()
    
    st.markdown("---")
    
    # Process button
    if st.button("🚀 Process Data", type="primary", use_container_width=True):
        if not st.session_state.db_connector:
            st.error("❌ Database not connected.")
        elif not st.session_state.selected_object:
            st.error("❌ No object selected.")
        else:
            with st.status("Processing fuel data...", expanded=True) as status:
                try:
                    obj = st.session_state.selected_object
                    start_dt = st.session_state.start_datetime
                    end_dt = st.session_state.end_datetime
                    
                    st.write(f"📋 Object: **{obj.object_label}** (Device: {obj.device_id})")
                    st.write(f"📅 Period: {start_dt} → {end_dt}")
                    
                    st.write("🔧 Running Algorithm 1: Sensor Data Processing...")
                    st.write("🔧 Running Algorithm 2: Event Detection...")
                    
                    fuel_df, events_df, summary, sensor_proc, events_proc = run_processing_pipeline(
                        st.session_state.db_connector,
                        obj, start_dt, end_dt,
                        st.session_state.sensor_config,
                        st.session_state.events_config
                    )
                    
                    st.session_state.processed_fuel_df = fuel_df
                    st.session_state.events_df = events_df
                    st.session_state.events = events_proc.events
                    st.session_state.data_gaps = sensor_proc.data_gaps
                    st.session_state.summary = summary
                    st.session_state.sensor_processor = sensor_proc  # Store for calibration viz
                    
                    st.write(f"✅ Processed **{len(fuel_df)}** data points")
                    st.write(f"⚡ Detected **{len(events_df)}** events")
                    
                    status.update(label="✅ Processing complete!", state="complete")
                    
                except ValueError as e:
                    status.update(label="❌ Data error", state="error")
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    status.update(label="❌ Processing failed", state="error")
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Show traceback"):
                        st.code(traceback.format_exc())
    
    # Configuration panel
    with st.expander("⚙️ Processing Configuration", expanded=False):
        render_config_panel()
    
    # Results
    if st.session_state.processed_fuel_df is not None:
        # Get measurement units from sensor processor
        sensor_proc = st.session_state.get('sensor_processor')
        measurement_units = getattr(sensor_proc, 'measurement_units', 'L') if sensor_proc else 'L'
        measurement_units_desc = getattr(sensor_proc, 'measurement_units_description', 'Liters') if sensor_proc else 'Liters'
        
        st.markdown("---")
        render_metrics(st.session_state.summary, measurement_units=measurement_units)
        
        st.markdown("---")
        col_title, col_option = st.columns([3, 1])
        with col_title:
            st.markdown(f"### 📈 Fuel Level Visualization ({measurement_units_desc})")
        with col_option:
            show_speed = st.checkbox("🚗 Show Speed", value=False)
        
        render_visualization(
            st.session_state.processed_fuel_df,
            st.session_state.events,
            st.session_state.data_gaps,
            show_speed=show_speed,
            measurement_units=measurement_units
        )
        
        # Calibration visualization (only if calibration was used)
        if st.session_state.processed_fuel_df["calibration_used"].any():
            with st.expander("🔬 Calibration: Raw Sensor → Calibrated Output", expanded=False):
                if hasattr(st.session_state, 'sensor_processor') and st.session_state.sensor_processor:
                    render_calibration_visualization(
                        st.session_state.processed_fuel_df,
                        st.session_state.sensor_processor
                    )
                else:
                    st.warning("⚠️ Sensor processor not available for calibration visualization.")
        
        st.markdown("### ⚡ Detected Events")
        render_events_table(st.session_state.events, measurement_units=measurement_units)
        
        st.markdown("---")
        st.markdown("### 💾 Silver Layer Output")
        render_silver_layer_tables(st.session_state.processed_fuel_df, st.session_state.events_df)


if __name__ == "__main__":
    main()
