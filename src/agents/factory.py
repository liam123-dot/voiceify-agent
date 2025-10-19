"""
Agent Factory for creating LiveKit agents from API configuration.

Converts agent data from the Voiceify API into LiveKit Agent instances.
"""

import logging
import asyncio
import aiohttp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
from livekit.agents import Agent, function_tool, RunContext
from livekit.agents.llm import ChatContext, ChatMessage


logger = logging.getLogger(__name__)


@dataclass
class AgentData:
    """
    Agent configuration data from the API.
    
    Attributes:
        id: Unique agent identifier
        name: Human-readable agent name
        configuration: Agent behavior configuration including instructions and pipeline
        tools: List of tools/functions available to the agent
        organization_id: Organization that owns this agent
        related_agents: Dict of related agents for transfers (agent_id -> AgentData)
        has_knowledge_base: Whether the agent has a knowledge base
    """
    
    id: str
    name: str
    configuration: Dict[str, Any]
    tools: Optional[List[Dict[str, Any]]] = None
    organization_id: Optional[str] = None
    related_agents: Optional[Dict[str, "AgentData"]] = None
    has_knowledge_base: bool = False
    
    @property
    def instructions(self) -> str:
        """Get agent instructions from configuration."""
        return self.configuration.get("instructions", "You are a helpful AI assistant.")
    
    @property
    def pipeline_type(self) -> str:
        """Get pipeline type from configuration."""
        return self.configuration.get("pipelineType", "pipeline")


class RAGAgent(Agent):
    """
    Agent with RAG (Retrieval-Augmented Generation) capabilities.
    
    This agent performs knowledge base lookups before generating responses,
    injecting relevant context into the conversation.
    """
    
    def __init__(
        self,
        agent_id: str,
        api_url: str,
        instructions: str,
        tools: Optional[List[Callable]] = None,
        event_callback: Optional[Callable] = None,
    ):
        """
        Initialize RAG-enabled agent.
        
        Args:
            agent_id: Agent ID for API lookup
            api_url: Base URL for the API
            instructions: Agent instructions
            tools: Optional list of tool functions
            event_callback: Optional callback for sending events to API
        """
        super().__init__(instructions=instructions, tools=tools)
        self.agent_id = agent_id
        self.api_url = api_url
        self.event_callback = event_callback
    
    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """
        Hook called after user completes their turn, before LLM generates response.
        
        Performs RAG lookup based on user's message and injects results into context.
        
        Args:
            turn_ctx: The chat context for this turn
            new_message: The user's latest message
        """
        try:
            # Extract the user's message text
            user_query = new_message.text_content
            
            if not user_query:
                logger.debug("No text content in user message, skipping RAG lookup")
                return
            
            logger.info(f"Performing RAG lookup for query: {user_query[:100]}...")
            
            # Fetch relevant context from knowledge base
            rag_content = await self._fetch_rag_context(user_query)

            print(rag_content)
            
            if rag_content:
                logger.info(f"Retrieved RAG context ({len(rag_content)} chars)")
                
                # Inject the context into the chat history
                turn_ctx.add_message(
                    role="assistant",
                    content=f"Relevant information from the knowledge base: {rag_content}"
                )
                
                # Send knowledge_retrieved event to API
                if self.event_callback:
                    try:
                        await self.event_callback(
                            "knowledge_retrieved",
                            {
                                "query": user_query,
                                "retrieved_context": rag_content,
                                "context_length": len(rag_content),
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error sending knowledge_retrieved event: {e}")
            else:
                logger.debug("No RAG content retrieved")
                
        except Exception as e:
            logger.error(f"Error in RAG lookup: {e}", exc_info=True)
            # Don't fail the entire turn if RAG fails, just log and continue
    
    async def _fetch_rag_context(self, query: str) -> Optional[str]:
        """
        Fetch relevant context from the knowledge base API.
        
        Args:
            query: User's query text
            
        Returns:
            Retrieved context string or None if fetch fails
        """
        try:
            endpoint = f"{self.api_url}/api/agents/{self.agent_id}/retrieve"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json={"query": query},
                    timeout=aiohttp.ClientTimeout(total=5.0),  # 5 second timeout
                ) as response:
                    if not response.ok:
                        error_text = await response.text()
                        logger.warning(f"RAG API returned {response.status}: {error_text}")
                        return None
                    
                    data = await response.json()
                    
                    # Extract the context from the response
                    # Adjust this based on your API response format
                    if isinstance(data, dict):
                        return data.get("context") or data.get("result") or data.get("data")
                    elif isinstance(data, str):
                        return data
                    else:
                        logger.warning(f"Unexpected RAG response format: {type(data)}")
                        return None
                        
        except asyncio.TimeoutError:
            logger.warning("RAG API request timed out")
            return None
        except Exception as e:
            logger.error(f"Error fetching RAG context: {e}")
            return None


def create_agent_from_config(
    agent_data: AgentData,
    tool_functions: Optional[List[Callable]] = None,
    api_url: Optional[str] = None,
    event_callback: Optional[Callable] = None,
) -> Agent:
    """
    Create a LiveKit Agent from API agent configuration.
    
    This function converts agent configuration from the Voiceify API format
    into a LiveKit Agent instance with the appropriate instructions and tools.
    If the agent has a knowledge base, creates a RAG-enabled agent.
    
    Tools are passed directly to the Agent constructor via the `tools` parameter.
    Each tool should already be decorated with @function_tool.
    
    Args:
        agent_data: Agent configuration from the API
        tool_functions: Optional list of tool functions (decorated with @function_tool)
        api_url: API base URL (required for RAG-enabled agents)
        event_callback: Optional callback for sending events to API
        
    Returns:
        Configured Agent instance (or RAGAgent if knowledge base enabled)
        
    Example:
        >>> agent_data = AgentData(
        ...     id="agent-123",
        ...     name="Customer Support",
        ...     configuration={"instructions": "Help customers with issues"},
        ...     has_knowledge_base=True
        ... )
        >>> agent = create_agent_from_config(agent_data, api_url="http://localhost:3000")
    """
    instructions = agent_data.instructions
    
    logger.info(f"Creating agent: {agent_data.name} ({agent_data.id})")
    logger.debug(f"  Instructions preview: {instructions[:100]}...")
    
    # Create appropriate agent type based on knowledge base configuration
    if agent_data.has_knowledge_base:
        if not api_url:
            logger.warning(f"Agent {agent_data.name} has knowledge base but no API URL provided, creating standard agent")
            if tool_functions:
                logger.info(f"  Adding {len(tool_functions)} tools to agent")
                agent = Agent(instructions=instructions, tools=tool_functions)
            else:
                agent = Agent(instructions=instructions)
        else:
            logger.info(f"  Creating RAG-enabled agent with knowledge base")
            if tool_functions:
                logger.info(f"  Adding {len(tool_functions)} tools to RAG agent")
            agent = RAGAgent(
                agent_id=agent_data.id,
                api_url=api_url,
                instructions=instructions,
                tools=tool_functions,
                event_callback=event_callback,
            )
    else:
        # Create standard Agent without RAG
        if tool_functions:
            logger.info(f"  Adding {len(tool_functions)} tools to agent")
            agent = Agent(instructions=instructions, tools=tool_functions)
        else:
            agent = Agent(instructions=instructions)
    
    # Store metadata for later use
    agent._voiceify_agent_id = agent_data.id
    agent._voiceify_agent_name = agent_data.name
    
    logger.info(f"Successfully created agent: {agent_data.name}")
    
    return agent

