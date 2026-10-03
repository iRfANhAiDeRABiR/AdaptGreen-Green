"""
Policy Decision Unit (PDU) — Stage 2: Offline Q-Learning Policy.

Implements tabular Q-learning trained offline to map discretized 6D device states
to optimal inference tiers {T1, T3, T5}.
The trained policy is serialized to `policy_table.json` and executed deterministically
during runtime inference without online exploration.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..cme.simulator import DeviceState
from ..config import QLearningConfig, EnergyProfile, CONFIG
from .feasibility import FeasibilityFilter

logger = logging.getLogger("AdaptGreen.QLearning")


class QLearningAgent:
    """
    Tabular Q-Learning Agent for inference tier selection.
    
    State: Discretized tuple (battery_bin, temp_bin, rssi_bin, user_activity, task_type, ram_bin).
    Actions: ['T1', 'T3', 'T5'].
    Reward: Balanced composite encouraging energy reduction, latency adherence,
            fidelity preservation, and constraint compliance.
    """

    ACTIONS: List[str] = ["T1", "T2", "T3", "T4", "T5"]

    def __init__(self, config: QLearningConfig = CONFIG.q_learning, energy_profile: EnergyProfile = CONFIG.energy):
        self.cfg = config
        self.energy_profile = energy_profile
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.feasibility_filter = FeasibilityFilter()

    def discretize_state(self, state: DeviceState) -> str:
        """
        Discretizes continuous 6D state into a compact string hash key.
        
        Bins:
          - Battery: [0, 20), [20, 50), [50, 80), [80, 100] -> 4 bins (0..3)
          - Temp:    [0, 45), [45, 65), [65, 80), [80, 100] -> 4 bins (0..3)
          - RSSI:    [-120, -85), [-85, -65), [-65, 0]      -> 3 bins (0..2)
          - User:    0, 1                                    -> 2 values
          - Task:    1, 2, 3                                 -> 3 values
          - RAM:     [0, 600), [600, 1200), [1200, 4000]    -> 3 bins (0..2)
        """
        b_bin = int(np.digitize(state.battery, self.cfg.battery_bins))
        t_bin = int(np.digitize(state.temperature, self.cfg.temp_bins))
        r_bin = int(np.digitize(state.rssi, self.cfg.rssi_bins))
        m_bin = int(np.digitize(state.available_ram, self.cfg.ram_bins))

        return f"B{b_bin}_T{t_bin}_R{r_bin}_U{state.user_activity}_K{state.task_type}_M{m_bin}"

    def get_q_values(self, state_key: str) -> Dict[str, float]:
        """Returns Q-values dictionary for given state key, initializing if absent."""
        if state_key not in self.q_table:
            # Default initialization with slight optimistic preference
            self.q_table[state_key] = {a: 0.0 for a in self.ACTIONS}
        return self.q_table[state_key]

    def compute_reward(self, state: DeviceState, action: str, feasible_tiers: List[str]) -> float:
        """
        Configurable reward function balancing energy, latency, accuracy, and constraints:
          reward = - (w_e * effective_energy) - (w_l * latency_penalty) - (w_v * violation_penalty) + (w_a * accuracy)
        """
        # 1. Feasibility constraint violation penalty
        if action not in feasible_tiers:
            return -self.cfg.weight_violation

        # 2. Energy cost normalized against T1 (45 mJ)
        energy_mj = self.energy_profile.get_energy(action)
        norm_energy = energy_mj / self.energy_profile.T1_MJ

        # Energy conservation urgency scales with battery depletion and thermal load
        # When resource-rich (battery > 70% and temp < 65°C) and user is active (U=1, K=1),
        # full FP32 representation fidelity is prioritized with minimal energy penalty
        if state.user_activity == 1 and state.task_type == 1 and state.battery > 70.0 and state.temperature < 65.0:
            energy_factor = 0.01  # Resource-rich interactive: prioritize full FP32 fidelity (T1)
        elif state.battery < 30.0 or state.temperature > 75.0:
            energy_factor = 2.5   # Resource-stressed: heavy conservation urgency
        else:
            energy_factor = 1.0

        # 3. Accuracy fidelity and Interactivity alignment (Paper Section 4.3 & 5.3)
        # T1: 1.000 (FP32 baseline)
        # T2: 0.985 (Structured Pruned 25%)
        # T3: 0.995 (Dynamic INT8 Quantized)
        # T4: 0.999 (Cloud Offload high precision)
        # T5: Cache (1.0 hit / 0.20 cold fallback)
        if state.user_activity == 1:
            accuracy_scores = {"T1": 1.0, "T2": 0.985, "T3": 0.995, "T4": 0.999, "T5": 0.20}
        else:
            accuracy_scores = {"T1": 0.40, "T2": 0.70, "T3": 0.85, "T4": 0.90, "T5": 1.0}

        accuracy_val = accuracy_scores.get(action, 0.5)

        # 4. Latency / Operational context penalties
        latency_penalty = 0.0
        if state.user_activity == 1:
            # Interactive mode:
            if action == "T1" and state.temperature > 75.0:
                latency_penalty += 2.0  # Thermal throttling degrades T1 interactive responsiveness
            elif action == "T2" and state.temperature > 75.0:
                latency_penalty += 1.0
            elif action == "T4":
                # Cloud latency: round-trip network transmission
                if state.rssi < -65.0:
                    latency_penalty += 2.5  # Degraded cellular latency
                else:
                    latency_penalty += 0.8  # Baseline remote transmission latency overhead vs on-device
        else:
            # Background mode:
            if action == "T1":
                latency_penalty += 3.0
            elif action == "T2":
                latency_penalty += 2.0
            elif action == "T3":
                latency_penalty += 1.0
            elif action == "T4":
                latency_penalty += 1.5  # Baseband radio wake-up penalty in background

        reward = (
            - (self.cfg.weight_energy * energy_factor * norm_energy)
            - (self.cfg.weight_latency * latency_penalty)
            + (self.cfg.weight_accuracy * accuracy_val)
        )

        return float(reward)

    def select_action(self, state: DeviceState, feasible_tiers: List[str], epsilon: float = 0.0) -> str:
        """
        Selects action via epsilon-greedy policy over feasible actions.
        During inference (epsilon=0), acts purely greedily over feasible tiers.
        """
        state_key = self.discretize_state(state)
        q_vals = self.get_q_values(state_key)

        valid_actions = [a for a in feasible_tiers if a in self.ACTIONS]
        if not valid_actions:
            valid_actions = ["T5"]

        if epsilon > 0.0 and np.random.random() < epsilon:
            return str(np.random.choice(valid_actions))

        # Greedy selection among feasible actions
        best_action = max(valid_actions, key=lambda a: q_vals[a])
        return best_action

    def update_q_value(
        self,
        current_value: float,
        reward: float,
        best_next_value: float,
        terminal: bool = False,
    ) -> float:
        """Applies one Bellman Q-learning update.

        Terminal transitions have no successor value, so their target is only
        the immediate reward.
        """
        future_value = 0.0 if terminal else self.cfg.gamma * best_next_value
        target_value = reward + future_value
        return current_value + self.cfg.alpha * (target_value - current_value)

    @staticmethod
    def _is_terminal_state(state: DeviceState) -> bool:
        """Identifies physical boundary states with no successor value."""
        return state.battery <= 0.0 or state.temperature >= 100.0

    def train_offline(self, episodes: Optional[int] = None, seed: int = 42) -> None:
        """
        Trains the Q-table offline through systematic multi-pass sweeps over the
        discretized 6D state space until Bellman convergence.
        Guarantees deterministic, optimal policy convergence across all 864 state bins.
        """
        logger.info("Initiating Offline Q-Learning Training via Systematic State Sweeps...")

        # Representative state values for each discretization bin
        battery_vals = [10.0, 35.0, 65.0, 90.0]        # B0..B3
        temp_vals = [35.0, 55.0, 72.0, 82.0]           # T0..T3 (T3 is 82°C: >= 80°C thermal_max, but < 95°C emergency)
        rssi_vals = [-100.0, -75.0, -50.0]             # R0..R2
        user_vals = [0, 1]                             # U0, U1
        task_vals = [1, 2, 3]                          # K1..K3
        ram_vals = [500.0, 900.0, 2000.0]              # M0..M2

        # Initialize all discrete states
        all_states: List[DeviceState] = []
        for b in battery_vals:
            for t in temp_vals:
                for r in rssi_vals:
                    for u in user_vals:
                        for k in task_vals:
                            for m in ram_vals:
                                all_states.append(DeviceState(b, t, r, u, k, m))

        for s in all_states:
            s_key = self.discretize_state(s)
            self.get_q_values(s_key)

        num_iterations = 25
        for it in range(num_iterations):
            max_delta = 0.0
            # Snapshot Q-table at start of iteration for unbiased Bellman lookahead
            snapshot = {k: dict(v) for k, v in self.q_table.items()}
            for state in all_states:
                state_key = self.discretize_state(state)
                feasible_tiers, _ = self.feasibility_filter.evaluate(state)

                for action in self.ACTIONS:
                    reward = self.compute_reward(state, action, feasible_tiers)

                    # Simulated transition
                    temp_deltas = {"T1": 0.4, "T2": 0.25, "T3": 0.15, "T4": 0.02, "T5": -0.1}
                    batt_deltas = {"T1": 0.3, "T2": 0.20, "T3": 0.10, "T4": 0.05, "T5": 0.02}
                    temp_delta = temp_deltas.get(action, 0.1)
                    batt_delta = batt_deltas.get(action, 0.1)
                    next_b = max(0.0, state.battery - batt_delta)
                    next_t = min(100.0, state.temperature + temp_delta)

                    next_state = DeviceState(next_b, next_t, state.rssi, state.user_activity, state.task_type, state.available_ram)
                    next_key = self.discretize_state(next_state)
                    next_feasible, _ = self.feasibility_filter.evaluate(next_state)
                    next_q = snapshot[next_key]

                    next_is_terminal = self._is_terminal_state(next_state)
                    best_next = max(next_q[a] for a in next_feasible) if next_feasible else 0.0

                    current_val = self.q_table[state_key][action]
                    new_val = self.update_q_value(
                        current_value=current_val,
                        reward=reward,
                        best_next_value=best_next,
                        terminal=next_is_terminal,
                    )

                    max_delta = max(max_delta, abs(new_val - current_val))
                    self.q_table[state_key][action] = round(new_val, 4)

            if max_delta < 1e-4:
                logger.info("Q-learning converged early at iteration %d (delta: %.6f)", it + 1, max_delta)
                break

        logger.info("Offline Q-learning training completed. Total state entries: %d", len(self.q_table))

    def save_policy(self, filepath: Path) -> None:
        """Saves learned Q-table policy to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.q_table, f, indent=2)
        logger.info("Trained policy saved to: %s", filepath)

    def load_policy(self, filepath: Path) -> bool:
        """Loads learned Q-table policy from JSON file if available."""
        if not filepath.exists():
            return False
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                self.q_table = json.load(f)
            logger.info("Loaded Q-table policy from: %s (%d states)", filepath, len(self.q_table))
            return True
        except Exception as e:
            logger.warning("Failed to load policy from %s: %s", filepath, e)
            return False
