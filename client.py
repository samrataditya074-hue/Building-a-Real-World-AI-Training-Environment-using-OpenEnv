# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Autonomous CEO Environment Client."""

from typing import Dict, Any

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import Action, Observation


class CEOEnv(EnvClient[Action, Observation, State]):
    """
    Client for the Autonomous CEO Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.
    """

    def _step_payload(self, action: Action) -> Dict[str, Any]:
        """
        Convert Action to JSON payload for step message.

        Args:
            action: Action instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        # Pydantic model can be easily dumped to a dictionary
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[Observation]:
        """
        Parse server response into StepResult[Observation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with Observation
        """
        obs_data = payload.get("observation", {})
        
        # OpenEnv servers return observation fields flattened and/or in metadata
        # We manually map them to the Observation model.
        observation = Observation(
            cash_norm=obs_data.get("cash_norm", 0.0),
            revenue_norm=obs_data.get("revenue_norm", 0.0),
            customer_satisfaction_norm=obs_data.get("customer_satisfaction_norm", 0.0),
            employee_morale_norm=obs_data.get("employee_morale_norm", 0.0),
            inventory_norm=obs_data.get("inventory_norm", 0.0),
            market_trend=obs_data.get("market_trend", 1.0),
            total_employees_norm=obs_data.get("total_employees_norm", 0.0),
            brand_reputation_norm=obs_data.get("brand_reputation_norm", 0.0),
            operational_efficiency_norm=obs_data.get("operational_efficiency_norm", 0.0),
            rd_progress_norm=obs_data.get("rd_progress_norm", 0.0),
            debt_norm=obs_data.get("debt_norm", 0.0),
            cash_crisis_flag=obs_data.get("cash_crisis_flag", 0.0),
            morale_crisis_flag=obs_data.get("morale_crisis_flag", 0.0),
            competitor_price_norm=obs_data.get("competitor_price_norm", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            info=payload.get("info", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
