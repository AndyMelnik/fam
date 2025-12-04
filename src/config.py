"""
Configuration module for Fuel Analytics
Defines Pydantic models for JSON config validation

Supports both:
- Legacy ProcessingConfig format (for backwards compatibility)
- New ETL Config format with metadata, sql_template, target, scheduler, security
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Processing Parameters Models (used in both old and new formats)
# =============================================================================

class HampelConfig(BaseModel):
    """Hampel filter configuration"""
    window_min_sec: int = Field(default=60, ge=10, le=3600)
    window_max_sec: int = Field(default=900, ge=60, le=7200)
    sigma_min: float = Field(default=2.0, ge=1.0, le=10.0)
    sigma_max: float = Field(default=4.0, ge=1.0, le=10.0)


class MedianConfig(BaseModel):
    """Rolling median filter configuration"""
    window_min_points: int = Field(default=3, ge=1, le=100)
    window_max_points: int = Field(default=15, ge=3, le=500)


class EMAConfig(BaseModel):
    """Exponential Moving Average configuration"""
    alpha_min: float = Field(default=0.05, ge=0.01, le=1.0)
    alpha_max: float = Field(default=0.5, ge=0.01, le=1.0)


class SmoothingConfig(BaseModel):
    """Combined smoothing configuration"""
    smoothing_level: int = Field(default=5, ge=1, le=10, description="Smoothing intensity from 1 (minimal) to 10 (maximum)")
    hampel: HampelConfig = Field(default_factory=HampelConfig)
    median: MedianConfig = Field(default_factory=MedianConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)


class ZeroHandlingConfig(BaseModel):
    """Zero value handling configuration"""
    mode: Literal["auto", "keep", "interpolate", "drop"] = "auto"
    startup_grace_period_min: int = Field(default=5, ge=0, le=60)
    min_zero_run_min: int = Field(default=10, ge=1, le=120)
    max_zero_level_l: float = Field(default=2.0, ge=0.0, le=10.0)


class RefuelDrainConfig(BaseModel):
    """Refuel/Drain detection thresholds"""
    min_step_l: float = Field(default=5.0, ge=0.1, le=100.0)
    min_step_pct_tank: float = Field(default=1.0, ge=0.1, le=50.0)
    min_step_pct_level: float = Field(default=5.0, ge=0.1, le=100.0)


class ClusterConfig(BaseModel):
    """Event clustering configuration"""
    merge_window_min_min: int = Field(default=5, ge=1, le=120)
    merge_window_min_max: int = Field(default=30, ge=5, le=240)
    min_points_in_cluster: int = Field(default=1, ge=1, le=100)
    # Plateau detection parameters (for accurate volume calculation)
    plateau_window_multiplier: float = Field(default=1.5, ge=1.0, le=5.0, 
        description="Multiplier for merge_window to find plateau regions")
    stability_threshold_l: float = Field(default=2.0, ge=0.1, le=10.0,
        description="Max std dev (liters) for stable plateau detection")


class EventConfig(BaseModel):
    """Final event filtering configuration"""
    min_volume_l: float = Field(default=10.0, ge=0.1, le=500.0)
    min_volume_pct_tank: float = Field(default=2.0, ge=0.1, le=100.0)
    allow_mixed_sign_cluster: bool = False


class DetectionConfig(BaseModel):
    """Event detection configuration"""
    mode: Literal["absolute_only", "relative_only", "absolute_and_relative"] = "absolute_and_relative"
    refuel: RefuelDrainConfig = Field(default_factory=RefuelDrainConfig)
    drain: RefuelDrainConfig = Field(default_factory=RefuelDrainConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    event: EventConfig = Field(default_factory=EventConfig)


class ContextConfig(BaseModel):
    """Context validation configuration"""
    max_valid_speed_kmh: float = Field(default=5.0, ge=0.0, le=1000.0)
    max_hdop: float = Field(default=5.0, ge=0.0, le=50.0)
    use_ignition: bool = True
    min_stop_duration_min: int = Field(default=2, ge=0, le=60)


# =============================================================================
# New ETL Config Models
# =============================================================================

class ChangeLogEntry(BaseModel):
    """Entry in the change log"""
    version: str
    date: str
    comment: str


class MetadataConfig(BaseModel):
    """ETL config metadata"""
    name: str = "Fuel ETL Config"
    description: str = ""
    model_name: str = "hampel_median_ema_delta_cluster_v1"
    model_version: str = "1.1.0"
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    created_by: str = "data_platform"
    tags: List[str] = Field(default_factory=lambda: ["fuel", "silver_layer"])
    owner_team: str = "data_platform"
    change_log: List[ChangeLogEntry] = Field(default_factory=list)


class SQLTemplateParameters(BaseModel):
    """SQL template parameters"""
    window_start: str = ":window_start"
    window_end: str = ":window_end"
    processing_version: str = "fuel_core_v1.1.0"


class SQLTemplateConfig(BaseModel):
    """SQL template configuration (for future Airflow integration)"""
    engine: str = "postgres"
    use_transaction: bool = True
    timeout_sec: int = Field(default=600, ge=30, le=7200)
    read_only_sources: bool = True
    query: str = " "  # Empty for now, Python logic is used
    parameters: SQLTemplateParameters = Field(default_factory=SQLTemplateParameters)


class PartitioningConfig(BaseModel):
    """Table partitioning configuration"""
    type: str = "time"
    column: str = "ts_utc"
    granularity: str = "day"


class TargetConfig(BaseModel):
    """Target table configuration"""
    schema_name: str = Field(default="silver_data_layer", alias="schema")
    table: str
    primary_keys: List[str]
    partitioning: PartitioningConfig = Field(default_factory=PartitioningConfig)
    indexes: List[List[str]] = Field(default_factory=list)
    retention_policy_days: int = Field(default=365, ge=1, le=3650)
    write_mode: Literal["insert", "upsert", "overwrite_partition"] = "upsert"
    idempotent: bool = True

    class Config:
        populate_by_name = True


class SchedulerConfig(BaseModel):
    """Scheduler configuration (for Airflow DAGs)"""
    cron_expression: str = "0 * * * *"
    enabled: bool = True
    max_runtime_sec: int = Field(default=1500, ge=60, le=86400)
    max_concurrent_runs: int = Field(default=1, ge=1, le=10)
    backfill_allowed: bool = False
    catchup_from: Optional[str] = None


class SecurityConfig(BaseModel):
    """Security configuration"""
    allowed_roles: List[str] = Field(default_factory=lambda: ["airflow", "data_platform"])
    source_schemas: List[str] = Field(default_factory=lambda: ["raw_telematics_data", "raw_business_data"])
    target_schema: str = "silver_data_layer"
    require_tls: bool = True
    no_secrets_in_config: bool = True


# =============================================================================
# Unified Parameters Config for ETL
# =============================================================================

class FuelSensorDataParameters(BaseModel):
    """Parameters for fuel_sensor_data ETL"""
    processing_window_hours: int = Field(default=48, ge=1, le=720)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    zero_handling: ZeroHandlingConfig = Field(default_factory=ZeroHandlingConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)


class FuelEventsParameters(BaseModel):
    """Parameters for fuel_events ETL"""
    processing_window_hours: int = Field(default=48, ge=1, le=720)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)


# =============================================================================
# Main ETL Config Classes
# =============================================================================

class FuelSensorDataETLConfig(BaseModel):
    """
    Complete ETL configuration for silver_data_layer.fuel_sensor_data
    
    Loaded from: config/fuel_sensor_data.etl.json
    """
    version: str = "1.0.0"
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    parameters: FuelSensorDataParameters = Field(default_factory=FuelSensorDataParameters)
    sql_template: SQLTemplateConfig = Field(default_factory=SQLTemplateConfig)
    target: TargetConfig = Field(default_factory=lambda: TargetConfig(
        table="fuel_sensor_data",
        primary_keys=["device_id", "ts_utc", "processing_version"]
    ))
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "FuelSensorDataETLConfig":
        """Load configuration from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, path: str | Path) -> None:
        """Save configuration to JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.model_dump(by_alias=True)

    def get_processing_version(self) -> str:
        """Get processing version string"""
        return self.sql_template.parameters.processing_version

    def get_adaptive_params(self, smoothing_level: Optional[int] = None) -> dict:
        """
        Calculate adaptive parameters based on smoothing_level.
        smoothing_level: 1 = minimal smoothing, 10 = maximum smoothing
        """
        level_raw = smoothing_level if smoothing_level is not None else self.parameters.smoothing.smoothing_level
        # Normalize to 0-1 range
        level = (level_raw - 1) / 9.0
        level = max(0.0, min(1.0, level))
        
        def lerp(min_val: float, max_val: float, t: float) -> float:
            return min_val + (max_val - min_val) * t

        return {
            "hampel_window_sec": lerp(
                self.parameters.smoothing.hampel.window_min_sec,
                self.parameters.smoothing.hampel.window_max_sec,
                level
            ),
            "hampel_sigma": lerp(
                self.parameters.smoothing.hampel.sigma_max,
                self.parameters.smoothing.hampel.sigma_min,
                level
            ),
            "median_window_points": int(lerp(
                self.parameters.smoothing.median.window_min_points,
                self.parameters.smoothing.median.window_max_points,
                level
            )),
            "ema_alpha": lerp(
                self.parameters.smoothing.ema.alpha_max,
                self.parameters.smoothing.ema.alpha_min,
                level
            ),
        }


class FuelEventsETLConfig(BaseModel):
    """
    Complete ETL configuration for silver_data_layer.fuel_events
    
    Loaded from: config/fuel_events.etl.json
    """
    version: str = "1.0.0"
    metadata: MetadataConfig = Field(default_factory=MetadataConfig)
    parameters: FuelEventsParameters = Field(default_factory=FuelEventsParameters)
    sql_template: SQLTemplateConfig = Field(default_factory=SQLTemplateConfig)
    target: TargetConfig = Field(default_factory=lambda: TargetConfig(
        table="fuel_events",
        primary_keys=["event_id"]
    ))
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "FuelEventsETLConfig":
        """Load configuration from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, path: str | Path) -> None:
        """Save configuration to JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.model_dump(by_alias=True)

    def get_processing_version(self) -> str:
        """Get processing version string"""
        return self.sql_template.parameters.processing_version

    def get_adaptive_cluster_params(self, smoothing_level: float = 0.5) -> dict:
        """
        Calculate adaptive clustering parameters.
        """
        def lerp(min_val: float, max_val: float, t: float) -> float:
            return min_val + (max_val - min_val) * t

        return {
            "merge_window_min": lerp(
                self.parameters.detection.cluster.merge_window_min_min,
                self.parameters.detection.cluster.merge_window_min_max,
                smoothing_level
            ),
        }


# =============================================================================
# Legacy ProcessingConfig (for backwards compatibility)
# =============================================================================

class ModelConfig(BaseModel):
    """Model metadata (legacy)"""
    name: str = "hampel_median_ema_delta_cluster_v1"
    smoothing_level: int = Field(default=5, ge=1, le=10, description="Smoothing intensity from 1 to 10")
    event_sensitivity: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str = "Hybrid filter + threshold detector"


class ProcessingConfig(BaseModel):
    """
    Main processing configuration (legacy format, maintained for compatibility).
    
    For new code, use FuelSensorDataETLConfig or FuelEventsETLConfig.
    """
    processing_version: str = "fuel_core_v1.1.0"
    model: ModelConfig = Field(default_factory=ModelConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    zero_handling: ZeroHandlingConfig = Field(default_factory=ZeroHandlingConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ProcessingConfig":
        """Load configuration from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_etl_configs(
        cls, 
        sensor_config: FuelSensorDataETLConfig, 
        events_config: FuelEventsETLConfig
    ) -> "ProcessingConfig":
        """
        Create ProcessingConfig from new ETL configs.
        Merges parameters from both configs for backwards compatibility.
        """
        return cls(
            processing_version=sensor_config.get_processing_version(),
            model=ModelConfig(
                name=sensor_config.metadata.model_name,
                smoothing_level=sensor_config.parameters.smoothing.smoothing_level,
                description=sensor_config.metadata.description
            ),
            smoothing=sensor_config.parameters.smoothing,
            zero_handling=sensor_config.parameters.zero_handling,
            detection=events_config.parameters.detection,
            context=events_config.parameters.context
        )

    def to_json_file(self, path: str | Path) -> None:
        """Save configuration to JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return self.model_dump()

    def get_adaptive_params(self, smoothing_level: Optional[int] = None) -> dict:
        """
        Calculate adaptive parameters based on smoothing_level.
        smoothing_level: 1 = minimal smoothing, 10 = maximum smoothing
        """
        level_raw = smoothing_level if smoothing_level is not None else self.smoothing.smoothing_level
        # Normalize to 0-1 range
        level = (level_raw - 1) / 9.0
        level = max(0.0, min(1.0, level))
        
        def lerp(min_val: float, max_val: float, t: float) -> float:
            return min_val + (max_val - min_val) * t

        return {
            "hampel_window_sec": lerp(
                self.smoothing.hampel.window_min_sec,
                self.smoothing.hampel.window_max_sec,
                level
            ),
            "hampel_sigma": lerp(
                self.smoothing.hampel.sigma_max,
                self.smoothing.hampel.sigma_min,
                level
            ),
            "median_window_points": int(lerp(
                self.smoothing.median.window_min_points,
                self.smoothing.median.window_max_points,
                level
            )),
            "ema_alpha": lerp(
                self.smoothing.ema.alpha_max,
                self.smoothing.ema.alpha_min,
                level
            ),
            "merge_window_min": lerp(
                self.detection.cluster.merge_window_min_min,
                self.detection.cluster.merge_window_min_max,
                level
            ),
        }


# =============================================================================
# Config Loaders
# =============================================================================

def get_config_dir() -> Path:
    """Get the config directory path"""
    return Path(__file__).parent.parent / "config"


def load_fuel_sensor_data_config() -> FuelSensorDataETLConfig:
    """Load fuel_sensor_data ETL configuration"""
    config_path = get_config_dir() / "fuel_sensor_data.etl.json"
    if config_path.exists():
        return FuelSensorDataETLConfig.from_json_file(config_path)
    return FuelSensorDataETLConfig()


def load_fuel_events_config() -> FuelEventsETLConfig:
    """Load fuel_events ETL configuration"""
    config_path = get_config_dir() / "fuel_events.etl.json"
    if config_path.exists():
        return FuelEventsETLConfig.from_json_file(config_path)
    return FuelEventsETLConfig()


def load_default_config() -> ProcessingConfig:
    """
    Load default configuration.
    
    First tries to load from new ETL configs, falls back to legacy format.
    """
    config_dir = get_config_dir()
    
    # Try new ETL configs first
    sensor_config_path = config_dir / "fuel_sensor_data.etl.json"
    events_config_path = config_dir / "fuel_events.etl.json"
    
    if sensor_config_path.exists() and events_config_path.exists():
        sensor_config = FuelSensorDataETLConfig.from_json_file(sensor_config_path)
        events_config = FuelEventsETLConfig.from_json_file(events_config_path)
        return ProcessingConfig.from_etl_configs(sensor_config, events_config)
    
    # Fallback to legacy config
    legacy_path = config_dir / "default_config.json"
    if legacy_path.exists():
        return ProcessingConfig.from_json_file(legacy_path)
    
    return ProcessingConfig()


def load_all_configs() -> tuple[FuelSensorDataETLConfig, FuelEventsETLConfig, ProcessingConfig]:
    """
    Load all configurations.
    
    Returns:
        Tuple of (sensor_config, events_config, legacy_config)
    """
    sensor_config = load_fuel_sensor_data_config()
    events_config = load_fuel_events_config()
    legacy_config = ProcessingConfig.from_etl_configs(sensor_config, events_config)
    
    return sensor_config, events_config, legacy_config
