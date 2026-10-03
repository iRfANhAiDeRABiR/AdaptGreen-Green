"""
Policy Decision Unit (PDU) — Main Policy Orchestrator.

Combines Stage 1 Feasibility Filter and Stage 2 Offline Q-Table Policy
with Hysteresis Cooldown (4 epochs) for tier-switching stability.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import logging

from ..cme.simulator import DeviceState
from ..config import SimulationConfig, CONFIG
from .feasibility import FeasibilityFilter
from .q_learning import QLearningAgent

logger = logging.getLogger("AdaptGreen.PDU")


@dataclass
class PolicyDecision:
    """Encapsulates the decision rendered by PDU for an epoch."""
    epoch: int
    selected_tier: str
    recommended_tier: str
    previous_tier: Optional[str]
    feasible_tiers: List[str]
    tier_switched: bool
    switch_reason: str
    cooldown_remaining: int
    is_emergency: bool


class PolicyDecisionUnit:
    """
    AdaptGreen Policy Decision Unit (PDU).
    Executes two-stage decision logic:
      Stage 1: Feasibility filter removes tiers violating constraints (C1-C6).
      Stage 2: Offline Q-learning policy selects optimal tier from feasible set.
    Governed by Hysteresis Cooldown to prevent ping-pong oscillation.
    """

    def __init__(self, config: SimulationConfig = CONFIG):
        self.cfg = config
        self.feasibility_filter = FeasibilityFilter(config.constraints)
        self.q_agent = QLearningAgent(config.q_learning, config.energy)

        # Hysteresis parameters
        self.cooldown_period: int = config.hysteresis.cooldown_epochs
        self.allow_emergency: bool = config.hysteresis.allow_emergency_fallback

        # Internal state tracking
        self.current_tier: Optional[str] = None
        self.cooldown_remaining: int = 0
        self.switch_history: List[Dict[str, Any]] = []

        # Load trained offline policy
        self._initialize_policy()

    def _initialize_policy(self) -> None:
        """Loads offline Q-table from policy_table.json or initiates training."""
        policy_path = self.cfg.policy_file
        if not self.q_agent.load_policy(policy_path):
            logger.info("Policy file %s not found. Training offline Q-learning policy...", policy_path)
            self.q_agent.train_offline(seed=self.cfg.random_seed)
            self.q_agent.save_policy(policy_path)

    def reset(self) -> None:
        """Resets runtime hysteresis state for a fresh simulation scenario."""
        self.current_tier = None
        self.cooldown_remaining = 0
        self.switch_history.clear()

    def decide(self, state: DeviceState, epoch: int) -> PolicyDecision:
        """
        Executes PDU decision pipeline for the given device state and epoch.

        Args:
            state: Current 6D mobile device state.
            epoch: Current simulation epoch index.

        Returns:
            PolicyDecision object detailing the selected tier and transition metadata.
        """
        # -------------------------------------------------------------
        # STAGE 1: Feasibility Filter
        # -------------------------------------------------------------
        feasible_tiers, violations = self.feasibility_filter.evaluate(state)

        # -------------------------------------------------------------
        # STAGE 2: Q-Table Policy Query
        # -------------------------------------------------------------
        recommended_tier = self.q_agent.select_action(state, feasible_tiers, epsilon=0.0)

        previous_tier = self.current_tier
        tier_switched = False
        switch_reason = "Normal execution"
        is_emergency = False

        # -------------------------------------------------------------
        # STAGE 3: Hysteresis and Stability Management
        # -------------------------------------------------------------
        if self.current_tier is None:
            # Initial epoch startup
            self.current_tier = recommended_tier
            self.cooldown_remaining = self.cooldown_period
            tier_switched = True
            switch_reason = f"Initial tier startup -> {recommended_tier}"
            self._log_switch(epoch, None, self.current_tier, switch_reason)

        else:
            # Check for catastrophic condition
            is_catastrophic = self.feasibility_filter.is_catastrophic_state(state)

            if is_catastrophic and self.allow_emergency and self.current_tier != "T5":
                # Emergency override: break hysteresis cooldown immediately
                is_emergency = True
                tier_switched = True
                switch_reason = "Emergency Fallback: Catastrophic environmental state detected"
                self._log_switch(epoch, self.current_tier, "T5", switch_reason)
                self.current_tier = "T5"
                self.cooldown_remaining = self.cooldown_period

            elif self.current_tier not in feasible_tiers:
                # Active tier has become infeasible! Must switch regardless of cooldown
                # Select the best available feasible tier recommended by Q-table
                fallback_tier = recommended_tier
                tier_switched = True
                active_violations = "; ".join(violations[self.current_tier])
                switch_reason = f"Feasibility constraint violation: {active_violations}"
                self._log_switch(epoch, self.current_tier, fallback_tier, switch_reason)
                self.current_tier = fallback_tier
                self.cooldown_remaining = self.cooldown_period

            elif self.cooldown_remaining > 0:
                # System is within hysteresis cooldown window; hold active tier
                self.cooldown_remaining -= 1
                tier_switched = False
                switch_reason = f"Hysteresis cooldown active ({self.cooldown_remaining} epochs remaining)"

            else:
                # Cooldown expired: check if policy recommends changing tier
                if recommended_tier != self.current_tier:
                    tier_switched = True
                    switch_reason = f"Q-table policy shift ({self.current_tier} -> {recommended_tier})"
                    self._log_switch(epoch, self.current_tier, recommended_tier, switch_reason)
                    self.current_tier = recommended_tier
                    self.cooldown_remaining = self.cooldown_period
                else:
                    tier_switched = False
                    switch_reason = f"Steady-state policy alignment with {self.current_tier}"

        return PolicyDecision(
            epoch=epoch,
            selected_tier=self.current_tier,
            recommended_tier=recommended_tier,
            previous_tier=previous_tier,
            feasible_tiers=feasible_tiers,
            tier_switched=tier_switched,
            switch_reason=switch_reason,
            cooldown_remaining=self.cooldown_remaining,
            is_emergency=is_emergency,
        )

    def _log_switch(self, epoch: int, from_tier: Optional[str], to_tier: str, reason: str) -> None:
        """Logs and records tier transition."""
        entry = {
            "epoch": epoch,
            "from_tier": from_tier,
            "to_tier": to_tier,
            "reason": reason,
        }
        self.switch_history.append(entry)
        from_str = from_tier if from_tier is not None else "None"
        print(f"\n[TIER SWITCH] Epoch {epoch:02d}: {from_str} -> {to_tier}")
        print(f"Reason: {reason}\n")
        logger.info("[TIER SWITCH] Epoch %02d: %s -> %s | Reason: %s", epoch, from_str, to_tier, reason)
