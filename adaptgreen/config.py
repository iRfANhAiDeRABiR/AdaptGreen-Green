"""
AdaptGreen Configuration Module.

Centralized configuration parameters for the Python simulation prototype of the
AdaptGreen context-adaptive runtime framework for energy-efficient mobile AI inference.
All thresholds, tier profiles, Q-learning hyper-parameters, and simulation paths
are maintained here to prevent hard-coding values across the codebase.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for AI workload models."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    max_seq_length: int = 128


@dataclass(frozen=True)
class EnergyProfile:
    """
    Simulated energy consumption profile per inference tier.
    Values specified by the AdaptGreen research paper:
      T1 (Full FP32): 45.0 mJ
      T2 (Structured Pruning): 30.0 mJ
      T3 (Dynamic INT8): 15.0 mJ
      T4 (Cloud Offload REST): 8.0 mJ
      T5 (In-memory Cache): 1.0 mJ
    """
    T1_MJ: float = 45.0
    T2_MJ: float = 30.0
    T3_MJ: float = 15.0
    T4_MJ: float = 8.0
    T5_MJ: float = 1.0

    def get_energy(self, tier: str) -> float:
        tier_map = {
            "T1": self.T1_MJ,
            "T2": self.T2_MJ,
            "T3": self.T3_MJ,
            "T4": self.T4_MJ,
            "T5": self.T5_MJ,
        }
        if tier not in tier_map:
            raise ValueError(f"Unknown tier: {tier}. Expected one of {list(tier_map.keys())}")
        return tier_map[tier]


@dataclass(frozen=True)
class MemoryProfile:
    """
    Simulated memory footprint (MB) per inference tier (Paper Section 4.3).
    """
    T1_MB: float = 120.0
    T2_MB: float = 85.0
    T3_MB: float = 50.0
    T4_MB: float = 15.0
    T5_MB: float = 5.0

    def get_memory(self, tier: str) -> float:
        tier_map = {
            "T1": self.T1_MB,
            "T2": self.T2_MB,
            "T3": self.T3_MB,
            "T4": self.T4_MB,
            "T5": self.T5_MB,
        }
        if tier not in tier_map:
            raise ValueError(f"Unknown tier: {tier}. Expected one of {list(tier_map.keys())}")
        return tier_map[tier]


@dataclass(frozen=True)
class FeasibilityConstraints:
    """
    Thresholds for Stage 1 Feasibility Filter (Constraints C1 - C6).
    
    C1: Latency deadline (ms) - Maximum tolerable inference delay for interactive tasks.
    C2: Accuracy degradation - Minimum acceptable cosine similarity fidelity relative to T1 FP32.
    C3: Thermal constraint (°C) - Max allowed SoC temperature before T1 is restricted.
    C4: Battery constraint (%) - Minimum battery level below which heavy tiers are restricted.
    C5: Network viability (dBm) - Minimum network RSSI for potential remote offloading.
    C6: Memory constraint (MB) - Minimum available RAM before memory-heavy operations are restricted.
    """
    thermal_max: float = 80.0          # °C (C3)
    battery_min: float = 20.0          # % (C4)
    ram_min: float = 600.0             # MB (C6)
    latency_max_ms: float = 120.0      # ms (C1)
    min_similarity_threshold: float = 0.95  # Cosine similarity to T1 (C2)
    network_rssi_min: float = -85.0    # dBm (C5)

    # Catastrophic condition thresholds for emergency fallback during cooldown
    catastrophic_thermal: float = 95.0 # °C (Emergency hardware protection)
    catastrophic_battery: float = 10.0 # %
    catastrophic_ram: float = 350.0    # MB


@dataclass(frozen=True)
class HysteresisConfig:
    """
    Hysteresis and tier-switching stability settings.
    Prevents rapid ping-pong oscillation between tiers.
    """
    cooldown_epochs: int = 4
    allow_emergency_fallback: bool = True


@dataclass(frozen=True)
class QLearningConfig:
    """
    Offline Q-learning hyper-parameters for Stage 2 PDU policy.
    Discretizes the 6D continuous device state into state indices.
    """
    alpha: float = 0.20              # Learning rate
    gamma: float = 0.9               # Discount factor for future-state value
    epsilon_start: float = 1.0       # Initial exploration rate
    epsilon_min: float = 0.05        # Final minimum exploration rate
    epsilon_decay: float = 0.995     # Exploration decay per episode
    episodes: int = 3500             # Number of training episodes

    # Reward weighting factors
    weight_energy: float = 1.2
    weight_violation: float = 12.0
    weight_latency: float = 0.6
    weight_accuracy: float = 2.0

    # Discretization bin edges for 6D state
    battery_bins: Tuple[float, ...] = (20.0, 50.0, 80.0)
    temp_bins: Tuple[float, ...] = (45.0, 65.0, 80.0)
    rssi_bins: Tuple[float, ...] = (-85.0, -65.0)
    ram_bins: Tuple[float, ...] = (600.0, 1200.0)


@dataclass(frozen=True)
class DeviceClassConfig:
    """
    Device class configuration bundle (Paper Section 5.3, 5.4, 6.1).
    Combines TierProfileTable and ConstraintConfig per device tier.
    """
    name: str                           # "Flagship", "Mid-range", "Budget"
    soc_description: str                # e.g. "Snapdragon 8 Gen", "Snapdragon 6-series", "MediaTek G-series"
    total_ram_mb: float                 # 12288.0, 6144.0, 4096.0
    ram_min: float                      # Safety headroom floor: 400.0, 600.0, 800.0 MB
    thermal_max: float                  # Thermal ceiling (°C): 82.0, 80.0, 78.0
    battery_min: float                  # Battery floor (%): 20.0
    energy_profile: EnergyProfile = field(default_factory=EnergyProfile)
    memory_profile: MemoryProfile = field(default_factory=MemoryProfile)


DEVICE_PRESETS: Dict[str, DeviceClassConfig] = {
    "FLAGSHIP": DeviceClassConfig(
        name="Flagship",
        soc_description="Snapdragon 8-series SoC",
        total_ram_mb=12288.0,
        ram_min=400.0,
        thermal_max=82.0,
        battery_min=20.0,
    ),
    "MID_RANGE": DeviceClassConfig(
        name="Mid-range",
        soc_description="Snapdragon 6-series SoC",
        total_ram_mb=6144.0,
        ram_min=600.0,
        thermal_max=80.0,
        battery_min=20.0,
    ),
    "BUDGET": DeviceClassConfig(
        name="Budget",
        soc_description="MediaTek G-series SoC",
        total_ram_mb=4096.0,
        ram_min=800.0,
        thermal_max=78.0,
        battery_min=20.0,
    ),
    "GALAXY_S23": DeviceClassConfig(
        name="Samsung Galaxy S23 Ultra",
        soc_description="Snapdragon 8 Gen 2 for Galaxy",
        total_ram_mb=12288.0,
        ram_min=400.0,
        thermal_max=83.0,
        battery_min=20.0,
        energy_profile=EnergyProfile(T1_MJ=38.0, T2_MJ=26.0, T3_MJ=12.5, T4_MJ=7.5, T5_MJ=0.9),
    ),
    "PIXEL_8": DeviceClassConfig(
        name="Google Pixel 8 Pro",
        soc_description="Google Tensor G3",
        total_ram_mb=12288.0,
        ram_min=500.0,
        thermal_max=81.0,
        battery_min=20.0,
        energy_profile=EnergyProfile(T1_MJ=42.0, T2_MJ=29.0, T3_MJ=14.0, T4_MJ=8.0, T5_MJ=1.0),
    ),
    "HELIO_G99": DeviceClassConfig(
        name="Redmi Note 12 / Helio G99",
        soc_description="MediaTek Helio G99 / Snapdragon 685",
        total_ram_mb=4096.0,
        ram_min=800.0,
        thermal_max=77.5,
        battery_min=15.0,
        energy_profile=EnergyProfile(T1_MJ=58.0, T2_MJ=40.0, T3_MJ=22.0, T4_MJ=9.5, T5_MJ=1.2),
    ),
}


@dataclass
class SimulationConfig:
    """Global configuration coordinating all simulation modules."""
    model: ModelConfig = field(default_factory=ModelConfig)
    energy: EnergyProfile = field(default_factory=EnergyProfile)
    memory: MemoryProfile = field(default_factory=MemoryProfile)
    constraints: FeasibilityConstraints = field(default_factory=FeasibilityConstraints)
    hysteresis: HysteresisConfig = field(default_factory=HysteresisConfig)
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)

    total_epochs: int = 60
    random_seed: int = 42
    polling_interval_ms: int = 500  # Conceptual Android polling, 1 decision step per epoch
    device_class: str = "MID_RANGE"

    # Directory layout
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    results_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    configs_dir: Path = field(init=False)
    policy_file: Path = field(init=False)
    papers_file: Path = field(init=False)

    def __post_init__(self):
        self.results_dir = self.base_dir / "results"
        self.figures_dir = self.results_dir / "figures"
        self.configs_dir = self.base_dir / "configs" / "devices"
        self.policy_file = self.base_dir / "pdu" / "policy_table.json"
        self.papers_file = self.base_dir / "data" / "paper_abstracts.json"

        # Ensure output directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

    def apply_device_class(self, device_class: str) -> None:
        """Applies hardware and constraint profiles for a target device class."""
        key = device_class.strip().upper().replace("-", "_").replace(" ", "_")
        
        # Check alias mappings
        aliases = {
            "SAMSUNG_GALAXY_S23": "GALAXY_S23",
            "SAMSUNG_GALAXY_S23_ULTRA": "GALAXY_S23",
            "GOOGLE_PIXEL_8": "PIXEL_8",
            "GOOGLE_PIXEL_8_PRO": "PIXEL_8",
            "BUDGET_HELIO_G99": "HELIO_G99",
            "REDMI_NOTE_12": "HELIO_G99"
        }
        key = aliases.get(key, key)

        if key not in DEVICE_PRESETS:
            # Check if json file exists directly
            json_file = self.configs_dir / f"{device_class.lower()}.json"
            if json_file.exists():
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                c = data.get("constraints", {})
                hw = data.get("hardware", {})
                tp = data.get("tier_profiles", {})
                custom_energy = EnergyProfile(
                    T1_MJ=tp.get("T1", {}).get("energy_mj", 45.0),
                    T2_MJ=tp.get("T2", {}).get("energy_mj", 30.0),
                    T3_MJ=tp.get("T3", {}).get("energy_mj", 15.0),
                    T4_MJ=tp.get("T4", {}).get("energy_mj", 8.0),
                    T5_MJ=tp.get("T5", {}).get("energy_mj", 1.0),
                )
                custom_memory = MemoryProfile(
                    T1_MB=tp.get("T1", {}).get("memory_mb", 120.0),
                    T2_MB=tp.get("T2", {}).get("memory_mb", 85.0),
                    T3_MB=tp.get("T3", {}).get("memory_mb", 50.0),
                    T4_MB=tp.get("T4", {}).get("memory_mb", 15.0),
                    T5_MB=tp.get("T5", {}).get("memory_mb", 5.0),
                )
                preset = DeviceClassConfig(
                    name=data.get("device_class", device_class),
                    soc_description=hw.get("soc", "Custom SoC"),
                    total_ram_mb=hw.get("total_ram_mb", 6144.0),
                    ram_min=c.get("ram_min", 600.0),
                    thermal_max=c.get("thermal_max", 80.0),
                    battery_min=c.get("battery_min", 20.0),
                    energy_profile=custom_energy,
                    memory_profile=custom_memory,
                )
            else:
                raise ValueError(f"Unknown device class: {device_class}. Supported: {list(DEVICE_PRESETS.keys())}")
        else:
            preset = DEVICE_PRESETS[key]

        self.device_class = key
        self.energy = preset.energy_profile
        self.memory = preset.memory_profile
        # Update constraints calibrated for this device class
        self.constraints = FeasibilityConstraints(
            thermal_max=preset.thermal_max,
            battery_min=preset.battery_min,
            ram_min=preset.ram_min,
        )


# Default shared instance
CONFIG = SimulationConfig()
