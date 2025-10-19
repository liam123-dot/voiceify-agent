"""
Agent Registry for managing pre-instantiated agents.

Stores agent instances to enable efficient agent-to-agent transfers without
recreating agents on each handoff.
"""

import logging
from typing import Dict, List, Optional
from livekit.agents import Agent


logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    Registry to store pre-instantiated agents for efficient transfers.
    
    This prevents the need to recreate agent instances during handoffs,
    which improves performance and maintains consistent state.
    
    Example:
        >>> registry = AgentRegistry()
        >>> registry.register("agent-123", my_agent)
        >>> agent = registry.get("agent-123")
        >>> if agent:
        ...     # Transfer to this agent
        ...     pass
    """
    
    def __init__(self):
        """Initialize an empty agent registry."""
        self._agents: Dict[str, Agent] = {}
        logger.debug("Agent registry initialized")
    
    def register(self, agent_id: str, agent: Agent) -> None:
        """
        Register an agent in the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            agent: Agent instance to store
        """
        self._agents[agent_id] = agent
        logger.info(f"Registered agent: {agent_id}")
    
    def get(self, agent_id: str) -> Optional[Agent]:
        """
        Get an agent from the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Agent instance if found, None otherwise
        """
        agent = self._agents.get(agent_id)
        if agent:
            logger.debug(f"Retrieved agent: {agent_id}")
        else:
            logger.warning(f"Agent not found in registry: {agent_id}")
        return agent
    
    def has(self, agent_id: str) -> bool:
        """
        Check if an agent exists in the registry.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            True if agent exists, False otherwise
        """
        return agent_id in self._agents
    
    def get_agent_ids(self) -> List[str]:
        """
        Get all registered agent IDs.
        
        Returns:
            List of agent IDs currently in the registry
        """
        return list(self._agents.keys())
    
    def size(self) -> int:
        """
        Get the number of registered agents.
        
        Returns:
            Count of agents in the registry
        """
        return len(self._agents)
    
    def clear(self) -> None:
        """Clear all agents from the registry."""
        count = len(self._agents)
        self._agents.clear()
        logger.info(f"Cleared {count} agents from registry")

