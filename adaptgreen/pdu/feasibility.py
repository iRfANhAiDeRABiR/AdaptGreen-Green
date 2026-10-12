"""
Policy Decision Unit (PDU) — Stage 1: Feasibility Filter.

Evaluates mobile device constraints (C1 through C6) against available inference tiers
[T1, T3, T5] to filter out non-viable tiers under current operational conditions.

Constraints Implemented:
  C1: Latency Constraint — Ensures inference latency meets application deadlines.
  C2: Accuracy Constraint — Preserves embedding fidelity relative to FP32 reference.
  C3: Thermal Constraint — Restricts high-power T1 when SoC temperature >= 80°C.
  C4: Battery Constraint — Restricts heavy tiers when battery level <= 20%.
  C5: Network Viability — Evaluates RSSI for potential network-assisted operations.
  C6: Memory Constraint — Restricts heavy models when available RAM < 600 MB.
"""

from typing import List, Dict, Tuple, Set
import logging
from ..cme.simulator import DeviceState
from ..config import FeasibilityConstraints, CONFIG

logger = logging.getLogger("AdaptGreen.FeasibilityFilter")


class FeasibilityFilter:
    """
    Stage 1 Feasibility Filter of the Policy Decision Unit.
    Filters candidate tiers {T1, T3, T5} based on environmental constraints.
    All thresholds are parameterized through FeasibilityConstraints.
    """

    ALL_TIERS: List[str] = ["T1", "T2", "T3", "T4", "T5"]
    # Expected fidelity values across 5 tiers (Paper Section 4.3 & 5.3)
    EXPECTED_ACCURACY: Dict[str, float] = {
        "T1": 1.0,
        "T2": 0.985,
        "T3": 0.995,
        "T4": 0.999,
    }

    def __init__(self, constraints: FeasibilityConstraints = CONFIG.constraints):
        self.constraints = constraints

    def evaluate(self, state: DeviceState) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Evaluates current DeviceState and filters out infeasible tiers across all 5 tiers.

        Args:
            state: Current 6D device state s(t).

        Returns:
            Tuple of:
              - feasible_tiers: Ordered list of allowed tiers.
              - violations: Dictionary mapping tier -> list of violated constraint descriptions.
        """
        violations: Dict[str, List[str]] = {tier: [] for tier in self.ALL_TIERS}

        # -------------------------------------------------------------
        # C3: Thermal Constraint (temperature >= thermal_max)
        # Discards high-computation tiers T1 and T2 when temperature >= 80.0°C
        # -------------------------------------------------------------
        if state.temperature >= self.constraints.thermal_max:
            msg = f"C3 Thermal: SoC temp {state.temperature:.1f}°C >= max threshold {self.constraints.thermal_max:.1f}°C"
            violations["T1"].append(msg)
            violations["T2"].append(msg)

        # Extreme catastrophic thermal runaway (>= 95°C): even T3 compute is unsafe
        if state.temperature >= self.constraints.catastrophic_thermal:
            violations["T3"].append(
                f"C3 Extreme Thermal: SoC temp {state.temperature:.1f}°C >= emergency threshold {self.constraints.catastrophic_thermal:.1f}°C"
            )

        # -------------------------------------------------------------
        # C4: Battery Constraint (battery <= battery_min)
        # Discards high-computation tiers T1 and T2 when battery <= 20.0%
        # -------------------------------------------------------------
        if state.battery <= self.constraints.battery_min:
            msg = f"C4 Battery: Battery level {state.battery:.1f}% <= threshold {self.constraints.battery_min:.1f}%"
            violations["T1"].append(msg)
            violations["T2"].append(msg)

        # Severe battery depletion (<= 10%): restrict T3 to conserve remaining charge
        if state.battery <= self.constraints.catastrophic_battery:
            violations["T3"].append(
                f"C4 Severe Battery: Battery level {state.battery:.1f}% <= emergency threshold {self.constraints.catastrophic_battery:.1f}%"
            )

        # -------------------------------------------------------------
        # C5: Network Viability Constraint (Paper Section 4.2)
        # Drops cloud offloading (T4) when network signal RSSI < -85 dBm
        # -------------------------------------------------------------
        if state.rssi < self.constraints.network_rssi_min:
            violations["T4"].append(
                f"C5 Network: RSSI {state.rssi:.1f} dBm < minimum signal floor {self.constraints.network_rssi_min:.1f} dBm"
            )

        # -------------------------------------------------------------
        # C6: Memory Headroom Constraint (available_ram < ram_min)
        # Eliminates tiers consuming RAM greater than available headroom
        # -------------------------------------------------------------
        if state.available_ram < self.constraints.ram_min:
            violations["T1"].append(
                f"C6 Memory: Available RAM {state.available_ram:.1f} MB < minimum required {self.constraints.ram_min:.1f} MB"
            )

        if state.available_ram < (self.constraints.ram_min - 100.0):
            violations["T2"].append(
                f"C6 Memory: Available RAM {state.available_ram:.1f} MB < T2 minimum required {self.constraints.ram_min - 100.0:.1f} MB"
            )

        if state.available_ram < self.constraints.catastrophic_ram:
            violations["T3"].append(
                f"C6 Critical Memory: Available RAM {state.available_ram:.1f} MB < critical threshold {self.constraints.catastrophic_ram:.1f} MB"
            )

        # -------------------------------------------------------------
        # C1: Latency Deadline Constraint
        # Evaluates interactive latency degradation under throttling
        # -------------------------------------------------------------
        if state.user_activity == 1 and state.temperature > 75.0 and state.available_ram < 800.0:
            violations["T1"].append(
                "C1 Latency: Thermal throttling and low RAM project unacceptable T1 interactive latency"
            )

        # -------------------------------------------------------------
        # C2: Accuracy Degradation Constraint
        # Checks if tier semantic fidelity meets minimum required floor
        # -------------------------------------------------------------
        for tier, expected_accuracy in self.EXPECTED_ACCURACY.items():
            if expected_accuracy < self.constraints.min_similarity_threshold:
                reason = (
                    f"C2 Accuracy: Expected fidelity {expected_accuracy:.3f} < "
                    f"minimum required {self.constraints.min_similarity_threshold:.3f}"
                )
                violations[tier].append(reason)
                logger.info("Rejected %s: %s", tier, reason)

        # Collect feasible tiers (tiers with no violations)
        feasible_tiers = [tier for tier in self.ALL_TIERS if len(violations[tier]) == 0]

        # Failsafe fallback: if all tiers violate, fallback to T5 cache or T3
        if not feasible_tiers:
            logger.warning("All tiers violated constraints! Engaging failsafe fallback to T5.")
            feasible_tiers = ["T5"]

        return feasible_tiers, violations

    def is_catastrophic_state(self, state: DeviceState) -> bool:
        """
        Determines whether the device state represents a critical emergency
        justifying breaking hysteresis cooldown to force minimum energy fallback (T5).
        """
        return (
            state.temperature >= self.constraints.catastrophic_thermal
            or state.battery <= self.constraints.catastrophic_battery
            or state.available_ram < self.constraints.catastrophic_ram
        )
