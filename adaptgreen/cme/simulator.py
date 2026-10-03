"""
Context Monitoring Engine (CME) Simulator.

Simulates the six-dimensional mobile device state:
  s(t) = (battery, temperature, network RSSI, user activity, task type, available RAM)

Generates dynamic environmental traces across four evaluation scenarios (ICCA 2026 Table 4):
  1. FOREGROUND_ACTIVE: Resource-rich interactive execution where T1 stays active for 60 epochs.
  2. BACKGROUND_BATCH: Display off, periodic background sync, prioritizing energy minimization.
  3. THERMAL_STRESS: Sustained compute until SoC hits 83°C, exercising C3 and hysteresis stability.
  4. BATTERY_CRITICAL: Elevated temperature, rapid battery depletion approaching 20%,
      and battery-driven background activity context for PDU decisions.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple
import numpy as np


@dataclass
class DeviceState:
    """
    Six-dimensional mobile device context state vector s(t).
    
    Attributes:
        battery: Battery level percentage [0.0, 100.0] %
        temperature: System-on-Chip (SoC) temperature [20.0, 100.0] °C
        rssi: Received Signal Strength Indication [-120.0, 0.0] dBm
        user_activity: Binary indicator {0, 1} where 1 = active foreground user interaction
        task_type: Categorical workload {1, 2, 3} (1 = latency-sensitive interactive query,
                   2 = periodic background sync, 3 = batch offline processing)
        available_ram: Available physical memory in Megabytes (MB)
    """
    battery: float
    temperature: float
    rssi: float
    user_activity: int
    task_type: int
    available_ram: float

    def to_dict(self) -> Dict[str, Any]:
        """Converts state attributes to a dictionary."""
        return asdict(self)

    def to_vector(self) -> List[float]:
        """Converts state attributes to a numeric feature vector."""
        return [
            float(self.battery),
            float(self.temperature),
            float(self.rssi),
            float(self.user_activity),
            float(self.task_type),
            float(self.available_ram),
        ]


class ContextMonitoringEngine:
    """
    Context Monitoring Engine (CME) responsible for periodic telemetry acquisition.
    In the simulation prototype, CME generates deterministic, reproducible state trajectories
    for specified scenario configurations.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def set_seed(self, seed: int) -> None:
        """Resets random number generator for scenario reproducibility."""
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate_state(self, scenario: str, epoch: int, total_epochs: int = 60) -> DeviceState:
        """
        Generates simulated device state for a given scenario and epoch.

        Args:
            scenario: Name of scenario ("FOREGROUND_ACTIVE" or "BATTERY_CRITICAL").
            epoch: Current simulation epoch (0-indexed, up to total_epochs - 1).
            total_epochs: Total number of simulation epochs (default 60).

        Returns:
            DeviceState representing the simulated environment at epoch t.
        """
        progress = float(epoch) / float(max(1, total_epochs - 1))  # Normalized [0.0, 1.0]

        normalized_scenario = scenario.strip().upper().replace(" ", "_")

        if normalized_scenario == "FOREGROUND_ACTIVE":
            return self._generate_foreground_active(epoch, progress)
        elif normalized_scenario == "BACKGROUND_BATCH":
            return self._generate_background_batch(epoch, progress)
        elif normalized_scenario == "THERMAL_STRESS":
            return self._generate_thermal_stress(epoch, progress)
        elif normalized_scenario == "BATTERY_CRITICAL":
            return self._generate_battery_critical(epoch, progress)
        elif normalized_scenario in ("NETWORK_ASSISTED_OFFLOAD", "CLOUD_OFFLOAD", "HYBRID_OFFLOAD"):
            return self._generate_network_assisted_offload(epoch, progress)
        else:
            raise ValueError(
                f"Unknown scenario: '{scenario}'. Supported: "
                "'FOREGROUND_ACTIVE', 'BACKGROUND_BATCH', 'THERMAL_STRESS', 'BATTERY_CRITICAL', 'NETWORK_ASSISTED_OFFLOAD'"
            )

    def _generate_foreground_active(self, epoch: int, progress: float) -> DeviceState:
        """
        Scenario 1: FOREGROUND_ACTIVE
        
        Conditions:
          - High battery gradually decreasing (98% down to ~86%)
          - Moderate temperature with gradual mild increase (40°C up to ~54°C, well below 80°C)
          - Abundant available RAM (~2200 MB to 2000 MB, well above 600 MB)
          - Strong Wi-Fi / 5G signal RSSI (~ -52 dBm)
          - Interactive foreground user workload: user_activity = 1, task_type = 1
        """
        # Smooth gradual battery decline with minor jitter
        base_battery = 98.0 - (progress * 12.0)
        battery = float(np.clip(base_battery + self.rng.normal(0, 0.2), 0.0, 100.0))

        # Modest SoC temperature increase under normal interactive load
        base_temp = 40.0 + (progress * 14.0)
        temperature = float(np.clip(base_temp + self.rng.normal(0, 0.3), 25.0, 70.0))

        # Stable high-quality network connection
        rssi = float(np.clip(-52.0 + self.rng.normal(0, 1.0), -120.0, 0.0))

        # Foreground interactive workload
        user_activity = 1
        task_type = 1

        # Healthy available RAM
        base_ram = 2200.0 - (progress * 200.0)
        available_ram = float(np.clip(base_ram + self.rng.normal(0, 15.0), 1000.0, 4000.0))

        return DeviceState(
            battery=round(battery, 2),
            temperature=round(temperature, 2),
            rssi=round(rssi, 2),
            user_activity=user_activity,
            task_type=task_type,
            available_ram=round(available_ram, 1),
        )

    def _generate_background_batch(self, epoch: int, progress: float) -> DeviceState:
        """
        Scenario 2: BACKGROUND_BATCH (ICCA 2026 Table 4)
        
        Conditions:
          - Display off, periodic background sync (user_activity = 0, task_type = 2)
          - Relaxes C1 latency ceiling, allowing PDU to prioritise energy minimisation
          - Moderate battery (92% draining gradually to ~82%)
          - Cool SoC temperature (36°C up to ~43°C, well below thermal thresholds)
          - Healthy available RAM (~1850 MB down to ~1750 MB)
          - Stable Wi-Fi network signal (~ -58 dBm)
        """
        base_battery = 92.0 - (progress * 10.0)
        battery = float(np.clip(base_battery + self.rng.normal(0, 0.2), 0.0, 100.0))

        base_temp = 36.0 + (progress * 7.0)
        temperature = float(np.clip(base_temp + self.rng.normal(0, 0.25), 20.0, 60.0))

        rssi = float(np.clip(-58.0 + self.rng.normal(0, 1.0), -120.0, 0.0))

        # Background batch mode: display off, periodic background sync
        user_activity = 0
        task_type = 2

        base_ram = 1850.0 - (progress * 100.0)
        available_ram = float(np.clip(base_ram + self.rng.normal(0, 10.0), 800.0, 4000.0))

        return DeviceState(
            battery=round(battery, 2),
            temperature=round(temperature, 2),
            rssi=round(rssi, 2),
            user_activity=user_activity,
            task_type=task_type,
            available_ram=round(available_ram, 1),
        )

    def _generate_thermal_stress(self, epoch: int, progress: float) -> DeviceState:
        """
        Scenario 3: THERMAL_STRESS (ICCA 2026 Table 4)
        
        Conditions:
          - Sustained active computation until SoC reaches 83°C
          - Validates C3 thermal constraint enforcement (80.0°C threshold)
          - Exercises hysteresis stability (cooldown) under repeated boundary crossings
          - Foreground interactive workload (user_activity = 1, task_type = 1)
          - Ample battery (90% down to ~65%, well above 20% C4 threshold)
          - Healthy RAM (~1700 MB down to ~1500 MB, well above 600 MB C6 threshold)
        """
        base_battery = 90.0 - (progress * 25.0)
        battery = float(np.clip(base_battery + self.rng.normal(0, 0.2), 0.0, 100.0))

        # Starts warm at ~60°C (bin T1, T1 active), ramps up under sustained load,
        # transitions to T3 near 65°C, crosses 80.0°C C3 boundary, and reaches peak at ~83.2°C
        base_temp = 60.0 + (progress * 23.0) + (0.35 * np.sin(progress * np.pi * 4))
        temperature = float(np.clip(base_temp + self.rng.normal(0, 0.15), 55.0, 84.0))

        rssi = float(np.clip(-52.0 + self.rng.normal(0, 1.0), -120.0, 0.0))

        user_activity = 1
        task_type = 1

        base_ram = 1700.0 - (progress * 200.0)
        available_ram = float(np.clip(base_ram + self.rng.normal(0, 12.0), 800.0, 4000.0))

        return DeviceState(
            battery=round(battery, 2),
            temperature=round(temperature, 2),
            rssi=round(rssi, 2),
            user_activity=user_activity,
            task_type=task_type,
            available_ram=round(available_ram, 1),
        )

    def _generate_battery_critical(self, epoch: int, progress: float) -> DeviceState:
        """
        Scenario 4: BATTERY_CRITICAL (ICCA 2026 Table 4)
        
        Conditions:
          - Starts warm but below the thermal threshold (~78°C < 80.0°C)
          - Rapidly depleting battery (starting at 75%, dropping below 20% down to ~15%)
          - Temperature rises across the run toward the elevated range, eventually engaging C3
                    - Application transitions from foreground (1) to background (0) as battery enters the critical range
        """
        # Rapid battery drop from 75% down to ~15% (crosses 20% near epoch 48)
        base_battery = 75.0 - (progress * 60.0)
        battery = float(np.clip(base_battery + self.rng.normal(0, 0.2), 5.0, 100.0))

        # Elevated temperature curve starting below thermal_max and rising under continuous load
        base_temp = 74.2 + (progress * 6.0) + (0.5 * np.sin(progress * np.pi * 2))
        temperature = float(np.clip(base_temp + self.rng.normal(0, 0.15), 78.0, 95.0))

        # Variable network RSSI
        rssi = float(np.clip(-70.0 - (progress * 15.0) + self.rng.normal(0, 1.2), -120.0, 0.0))

        # Background work begins after ~70% of the epochs (battery <= 32.0%)
        # Strictly reproduces ICCA 2026 Table 5: 43 epochs T3 (0..42) and 17 epochs T5 (43..59)
        is_background = battery <= 32.0
        user_activity = 0 if is_background else 1
        task_type = 2 if is_background else 1

        # Available RAM drops under system memory pressure
        base_ram = 1100.0 - (progress * 550.0)
        available_ram = float(np.clip(base_ram + self.rng.normal(0, 10.0), 400.0, 2000.0))

        return DeviceState(
            battery=round(battery, 2),
            temperature=round(temperature, 2),
            rssi=round(rssi, 2),
            user_activity=user_activity,
            task_type=task_type,
            available_ram=round(available_ram, 1),
        )

    def _generate_network_assisted_offload(self, epoch: int, progress: float) -> DeviceState:
        """
        Scenario 5: NETWORK_ASSISTED_OFFLOAD (T3 -> T4 -> T5 Transition)

        Demonstrates dynamic 3-phase tier migration:
          - Phase 1 (Epochs 0..19): Moderate battery (45% -> 35%), low RAM (520 MB < 600 MB C6 floor),
            weak cellular network (-75 dBm < -65 dBm R1 bin), active interactive query workload.
            Result: Runs energy-efficient on-device INT8 (T3).
          - Phase 2 (Epochs 20..39): Battery drops into critical zone (<20%), connects to strong high-speed
            Wi-Fi (RSSI = -50 dBm >= -65 dBm R2 bin), active interactive queries continue.
            Result: Switches from on-device compute to Cloud Offload (T4) to preserve dying battery.
          - Phase 3 (Epochs 40..59): Screen turns off (user_activity = 0), shifts to periodic background
            refresh (task_type = 2), recurring queries hit the embedding cache.
            Result: Switches from Cloud Offload to In-Memory Cache (T5) for near-zero energy consumption.
        """
        if progress < 0.333:
            # Phase 1: On-device INT8 (T3)
            base_battery = 45.0 - (progress * 30.0)
            battery = float(np.clip(base_battery + self.rng.normal(0, 0.2), 30.0, 50.0))
            temperature = float(np.clip(48.0 + (progress * 6.0) + self.rng.normal(0, 0.2), 40.0, 60.0))
            rssi = float(np.clip(-75.0 + self.rng.normal(0, 1.0), -90.0, -70.0))
            user_activity = 1
            task_type = 1
            available_ram = float(np.clip(520.0 + self.rng.normal(0, 5.0), 450.0, 580.0))
        elif progress < 0.667:
            # Phase 2: Cloud Offload (T4)
            p_phase2 = (progress - 0.333) / 0.334
            base_battery = 18.0 - (p_phase2 * 4.0)
            battery = float(np.clip(base_battery + self.rng.normal(0, 0.15), 14.0, 19.5))
            temperature = float(np.clip(42.0 - (p_phase2 * 3.0) + self.rng.normal(0, 0.2), 35.0, 48.0))
            rssi = float(np.clip(-50.0 + self.rng.normal(0, 0.8), -60.0, -40.0))
            user_activity = 1
            task_type = 1
            available_ram = float(np.clip(520.0 + self.rng.normal(0, 5.0), 450.0, 580.0))
        else:
            # Phase 3: In-Memory Cache (T5)
            p_phase3 = (progress - 0.667) / 0.333
            base_battery = 14.0 - (p_phase3 * 3.0)
            battery = float(np.clip(base_battery + self.rng.normal(0, 0.1), 10.5, 14.0))
            temperature = float(np.clip(38.0 - (p_phase3 * 4.0) + self.rng.normal(0, 0.2), 30.0, 42.0))
            rssi = float(np.clip(-50.0 + self.rng.normal(0, 0.8), -60.0, -40.0))
            user_activity = 0
            task_type = 2
            available_ram = float(np.clip(520.0 + self.rng.normal(0, 5.0), 450.0, 580.0))

        return DeviceState(
            battery=round(battery, 2),
            temperature=round(temperature, 2),
            rssi=round(rssi, 2),
            user_activity=user_activity,
            task_type=task_type,
            available_ram=round(available_ram, 1),
        )



class LumpedRCThermalModel:
    """
    Two-Node Lumped Parameter RC Thermal Circuit Model.
    
    Models heat flow across mobile silicon die, chassis skin, and ambient environment
    using differential Fourier heat transfer equations:
      C_die  * (dT_die/dt)  = P_active - (T_die - T_skin) / R_die_skin
      C_skin * (dT_skin/dt) = (T_die - T_skin) / R_die_skin - (T_skin - T_amb) / R_skin_amb

    Attributes:
        t_die: Silicon junction die temperature (°C)
        t_skin: Outer device chassis skin temperature (°C)
        t_ambient: Ambient environment temperature (°C)
        c_die: Thermal capacitance of silicon die (J/K)
        c_skin: Thermal capacitance of chassis/skin (J/K)
        r_die_skin: Thermal resistance from die to skin (K/W)
        r_skin_amb: Thermal resistance from skin to ambient (K/W)
    """

    def __init__(
        self,
        t_ambient: float = 25.0,
        initial_t_die: float = 38.0,
        initial_t_skin: float = 30.0,
        c_die: float = 0.12,
        c_skin: float = 18.0,
        r_die_skin: float = 4.2,
        r_skin_amb: float = 14.5,
    ):
        self.t_ambient = t_ambient
        self.t_die = initial_t_die
        self.t_skin = initial_t_skin
        self.c_die = c_die
        self.c_skin = c_skin
        self.r_die_skin = r_die_skin
        self.r_skin_amb = r_skin_amb

        # Average active power dissipation per tier in Watts
        self.tier_power_map = {
            "T1": 4.80,   # FP32 MiniLM heavy compute
            "T2": 3.20,   # Structured pruning moderate compute
            "T3": 1.60,   # INT8 quantized efficient compute
            "T4": 0.85,   # Radio transmission
            "T5": 0.15,   # Memory cache lookup
            "IDLE": 0.35  # Device baseline idle power
        }

    def reset(self, t_die: float = 38.0, t_skin: float = 30.0) -> None:
        """Resets die and skin temperatures."""
        self.t_die = t_die
        self.t_skin = t_skin

    def step(self, active_power_watts: float, delta_time_s: float = 1.0) -> Tuple[float, float]:
        """
        Integrates heat flow across delta_time_s using Euler discretization.

        Args:
            active_power_watts: Heat generated by processor during time step.
            delta_time_s: Time step duration in seconds.

        Returns:
            Tuple of (t_die_new, t_skin_new).
        """
        # Sub-step integration for high numerical stability (10 sub-steps per delta_t)
        sub_steps = 10
        sub_dt = delta_time_s / sub_steps

        for _ in range(sub_steps):
            q_die_to_skin = (self.t_die - self.t_skin) / self.r_die_skin
            q_skin_to_amb = (self.t_skin - self.t_ambient) / self.r_skin_amb

            dt_die = (active_power_watts - q_die_to_skin) / self.c_die * sub_dt
            dt_skin = (q_die_to_skin - q_skin_to_amb) / self.c_skin * sub_dt

            self.t_die += dt_die
            self.t_skin += dt_skin

        return float(round(self.t_die, 2)), float(round(self.t_skin, 2))

    def step_tier(self, tier: str, delta_time_s: float = 1.0) -> Tuple[float, float]:
        """Steps thermal dynamics driven by the active tier's power dissipation."""
        norm_tier = tier.strip().upper() if tier else "IDLE"
        power_w = self.tier_power_map.get(norm_tier, 1.5)
        return self.step(power_w, delta_time_s=delta_time_s)

    def simulate_trajectory(self, tier_sequence: List[str], dt_per_step: float = 1.0) -> List[Dict[str, float]]:
        """Simulates full thermal trajectory for a given sequence of tier actions."""
        history = []
        for tier in tier_sequence:
            t_die, t_skin = self.step_tier(tier, delta_time_s=dt_per_step)
            history.append({
                "tier": tier,
                "temperature_die_c": t_die,
                "temperature_skin_c": t_skin,
            })
        return history

