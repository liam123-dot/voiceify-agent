"""
Agent Factory for creating LiveKit agents from API configuration.

Converts agent data from the Voiceify API into LiveKit Agent instances.
"""

import os
import logging
import asyncio
import aiohttp
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Tuple
from livekit.agents import Agent, function_tool, RunContext
from livekit.agents.llm import ChatContext, ChatMessage

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Supabase SDK not installed. Install with: pip install supabase")

# Commented out - switched to Voyage AI embeddings
# try:
#     from openai import OpenAI
#     OPENAI_AVAILABLE = True
# except ImportError:
#     OPENAI_AVAILABLE = False
#     logger = logging.getLogger(__name__)
#     logger.warning("OpenAI SDK not installed. Install with: pip install openai")


logger = logging.getLogger(__name__)


class KnowledgeBaseRetriever:
    """
    Handles knowledge base queries for both RAG and tool modes.
    
    This class encapsulates the logic for querying knowledge bases using
    Voyage AI embeddings and Supabase pgvector similarity search.
    """
    
    def __init__(
        self,
        supabase_client: "Client",
        voyage_api_key: str,
        knowledge_base_ids: List[str],
        event_callback: Optional[Callable] = None,
    ):
        """
        Initialize the knowledge base retriever.
        
        Args:
            supabase_client: Supabase client instance
            voyage_api_key: Voyage AI API key for embeddings
            knowledge_base_ids: List of knowledge base IDs to search
            event_callback: Optional callback for sending events to API
        """
        logger.info("🔍 Initializing KnowledgeBaseRetriever")
        logger.info(f"  Knowledge base IDs: {knowledge_base_ids}")
        logger.info(f"  Number of knowledge bases: {len(knowledge_base_ids)}")
        logger.info(f"  Event callback enabled: {event_callback is not None}")
        
        self.supabase = supabase_client
        self.voyage_api_key = voyage_api_key
        self.knowledge_base_ids = knowledge_base_ids
        self.event_callback = event_callback
        # Track pending RAG events for pre-injection mode
        self.pending_rag_events: List[Dict] = []
        
        logger.info("✅ KnowledgeBaseRetriever initialized successfully")
    
    async def query_knowledge_base(
        self,
        query: str,
        speech_id: Optional[str] = None
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Query knowledge base and return context + metrics.
        
        Args:
            query: User's query text
            speech_id: Optional speech ID for correlation (used in pre-injection mode)
            
        Returns:
            Tuple of (context_string or None, metrics_dict)
            
            metrics_dict contains:
                - latency_ms: Total latency in milliseconds
                - embedding_latency_ms: Embedding generation latency
                - supabase_query_latency_ms: Supabase query latency
                - context_length: Length of retrieved context (if any)
        """
        start_time = time.time()
        
        # If no knowledge bases are assigned, return None
        if not self.knowledge_base_ids:
            logger.debug("No knowledge bases assigned")
            latency_ms = int((time.time() - start_time) * 1000)
            return None, {
                "latency_ms": latency_ms,
                "embedding_latency_ms": 0,
                "supabase_query_latency_ms": 0,
            }
        
        try:
            # Generate embedding for the query using Voyage AI
            logger.info(f"Generating embedding for query: {query[:100]}...")
            
            # Call Voyage AI embeddings API via HTTP
            embedding_start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    'https://api.voyageai.com/v1/embeddings',
                    headers={
                        'Authorization': f'Bearer {self.voyage_api_key}',
                        'Content-Type': 'application/json',
                    },
                    json={
                        'input': [query.strip()],
                        'model': 'voyage-3.5-lite',
                    }
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Voyage AI API error {response.status}: {error_text}")
                        raise ValueError(f"Voyage AI API returned status {response.status}")
                    
                    embedding_response = await response.json()
            
            query_embedding = embedding_response['data'][0]['embedding']
            embedding_time_ms = int((time.time() - embedding_start) * 1000)
            logger.info(f"Generated embedding with {len(query_embedding)} dimensions in {embedding_time_ms}ms")
            
            # Call the Supabase RPC function for similarity search across all knowledge bases
            logger.info(f"Searching {len(self.knowledge_base_ids)} knowledge base(s): {self.knowledge_base_ids}")
            logger.info(f"Using RPC function 'match_knowledge_base_documents' with threshold=0, count=3")
            
            # Run in thread pool as Supabase client is synchronous
            loop = asyncio.get_event_loop()
            query_start = time.time()
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: self.supabase.rpc(
                        'match_knowledge_base_documents',
                        {
                            'query_embedding': query_embedding,
                            'kb_ids': self.knowledge_base_ids,
                            'match_threshold': 0,  # Lower threshold for debugging
                            'match_count': 3
                        }
                    ).execute()
                )
                query_time_ms = int((time.time() - query_start) * 1000)
                total_time_ms = int((time.time() - start_time) * 1000)
                
                logger.info(f"RPC call successful in {query_time_ms}ms. Response data type: {type(response.data)}")
                logger.info(f"RPC returned {len(response.data) if response.data else 0} documents")
                logger.info(f"Timing - Embedding: {embedding_time_ms}ms, Query: {query_time_ms}ms, Total: {total_time_ms}ms")
                
                if response.data and len(response.data) > 0:
                    # Log first result for debugging
                    logger.info(f"First result keys: {list(response.data[0].keys())}")
                    logger.info(f"First result similarity: {response.data[0].get('similarity')}")
                    
            except Exception as e:
                logger.error(f"Error calling RPC function: {e}", exc_info=True)
                latency_ms = int((time.time() - start_time) * 1000)
                return None, {
                    "latency_ms": latency_ms,
                    "embedding_latency_ms": embedding_time_ms,
                    "supabase_query_latency_ms": 0,
                }
            
            latency_ms = total_time_ms
            
            if not response.data:
                logger.warning("No documents retrieved from knowledge bases")
                return None, {
                    "latency_ms": latency_ms,
                    "embedding_latency_ms": embedding_time_ms,
                    "supabase_query_latency_ms": query_time_ms,
                }
            
            # Extract content from returned documents (already sorted by similarity)
            context_parts = [doc.get('content', '') for doc in response.data if doc.get('content')]
            
            logger.info(f"Extracted {len(context_parts)} content parts from {len(response.data)} documents")
            
            if not context_parts:
                logger.debug("No content in retrieved documents")
                return None, {
                    "latency_ms": latency_ms,
                    "embedding_latency_ms": embedding_time_ms,
                    "supabase_query_latency_ms": query_time_ms,
                }
            
            # Join all context parts
            context = "\n\n".join(context_parts)
            logger.info(f"Retrieved {len(context_parts)} documents (total {len(context)} chars)")
            
            return context, {
                "latency_ms": latency_ms,
                "embedding_latency_ms": embedding_time_ms,
                "supabase_query_latency_ms": query_time_ms,
                "context_length": len(context),
            }
            
        except Exception as e:
            logger.error(f"Error fetching knowledge base context: {e}", exc_info=True)
            latency_ms = int((time.time() - start_time) * 1000)
            return None, {
                "latency_ms": latency_ms,
                "embedding_latency_ms": 0,
                "supabase_query_latency_ms": 0,
            }


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
        knowledge_base_ids: List of knowledge base IDs assigned to this agent (empty list if none)
    """
    
    id: str
    name: str
    configuration: Dict[str, Any]
    tools: Optional[List[Dict[str, Any]]] = None
    organization_id: Optional[str] = None
    related_agents: Optional[Dict[str, "AgentData"]] = None
    knowledge_base_ids: List[str] = None
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.knowledge_base_ids is None:
            self.knowledge_base_ids = []
    
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
    injecting relevant context into the conversation using Supabase pgvector.
    """
    
    def __init__(
        self,
        agent_id: str,
        organization_id: str,
        knowledge_base_ids: List[str],
        instructions: str,
        tools: Optional[List[Callable]] = None,
        event_callback: Optional[Callable] = None,
    ):
        """
        Initialize RAG-enabled agent.
        
        Args:
            agent_id: Agent ID for API lookup
            organization_id: Organization ID
            knowledge_base_ids: List of knowledge base IDs to search
            instructions: Agent instructions
            tools: Optional list of tool functions
            event_callback: Optional callback for sending events to API
        """
        logger.info("🤖 Initializing RAGAgent (Pre-injection mode)")
        logger.info(f"  Agent ID: {agent_id}")
        logger.info(f"  Organization ID: {organization_id}")
        logger.info(f"  Knowledge bases: {knowledge_base_ids}")
        logger.info(f"  Number of tools: {len(tools) if tools else 0}")
        
        super().__init__(instructions=instructions, tools=tools)
        self.agent_id = agent_id
        self.organization_id = organization_id
        self.event_callback = event_callback
        
        # Initialize Supabase client
        logger.info("  Setting up Supabase client for knowledge base queries")
        supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.error("NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables not set")
            raise ValueError("NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for RAG-enabled agents")
        
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase SDK is not installed. Install with: pip install supabase")
        
        supabase_client: Client = create_client(supabase_url, supabase_key)
        logger.info("  ✓ Supabase client created")
        
        # Initialize Voyage AI API key for embeddings
        logger.info("  Setting up Voyage AI for embeddings")
        voyage_api_key = os.getenv("VOYAGE_API_KEY")
        if not voyage_api_key:
            logger.error("VOYAGE_API_KEY environment variable not set")
            raise ValueError("VOYAGE_API_KEY is required for RAG-enabled agents")
        logger.info("  ✓ Voyage AI configured")
        
        # Create knowledge base retriever
        logger.info("  Creating KnowledgeBaseRetriever...")
        self.retriever = KnowledgeBaseRetriever(
            supabase_client=supabase_client,
            voyage_api_key=voyage_api_key,
            knowledge_base_ids=knowledge_base_ids,
            event_callback=event_callback,
        )
        
        logger.info("✅ RAGAgent initialized successfully")
        
        # Commented out - switched to Voyage AI embeddings
        # # Initialize OpenAI client for embeddings
        # openai_api_key = os.getenv("OPENAI_API_KEY")
        # if not openai_api_key:
        #     logger.error("OPENAI_API_KEY environment variable not set")
        #     raise ValueError("OPENAI_API_KEY is required for RAG-enabled agents")
        # 
        # if not OPENAI_AVAILABLE:
        #     raise ImportError("OpenAI SDK is not installed. Install with: pip install openai")
        # 
        # self.openai_client = OpenAI(api_key=openai_api_key)
    
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
            logger.info("📚 on_user_turn_completed called - RAG lookup triggered")
            
            # Extract the user's message text
            user_query = new_message.text_content
            
            if not user_query:
                logger.warning("  No text content in user message, skipping RAG lookup")
                return
            
            logger.info(f"  User query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
            logger.info(f"  Query length: {len(user_query)} characters")
            logger.info("  Starting knowledge base query...")
            
            # Fetch relevant context from knowledge base using retriever
            rag_content, metrics = await self.retriever.query_knowledge_base(user_query)

            print(rag_content)
            
            if rag_content:
                logger.info(f"✅ Knowledge base query successful!")
                logger.info(f"  Retrieved context length: {len(rag_content)} chars")
                logger.info(f"  Total latency: {metrics['latency_ms']}ms")
                logger.info(f"  Breakdown - Embedding: {metrics['embedding_latency_ms']}ms, Query: {metrics['supabase_query_latency_ms']}ms")
                logger.info(f"  Injecting context into conversation...")
                
                # Inject the context into the chat history
                turn_ctx.add_message(
                    role="assistant",
                    content=f"Relevant information from the knowledge base: {rag_content}"
                )
                logger.info("  ✓ Context injected into chat history")
                
                # Store RAG event data for later correlation with speech ID
                rag_event_data = {
                    "query": user_query,
                    "retrieved_context": rag_content,
                    "context_length": len(rag_content),
                    "latency_ms": metrics["latency_ms"],
                    "embedding_latency_ms": metrics["embedding_latency_ms"],
                    "supabase_query_latency_ms": metrics["supabase_query_latency_ms"],
                    "timestamp": time.time(),
                }
                
                # Store as pending event to be correlated with speech ID later
                self.retriever.pending_rag_events.append(rag_event_data)
                logger.info(f"  ✓ Stored pending RAG event for speech correlation")
                logger.info(f"  Pending RAG events: {len(self.retriever.pending_rag_events)}")
                
                # Also send immediate event without speech ID for backwards compatibility
                if self.event_callback:
                    try:
                        logger.info("  Sending knowledge_retrieved event to API...")
                        await self.event_callback(
                            "knowledge_retrieved",
                            rag_event_data
                        )
                        logger.info("  ✓ Event sent successfully")
                    except Exception as e:
                        logger.error(f"  ❌ Error sending knowledge_retrieved event: {e}")
            else:
                logger.warning("⚠️  No relevant content found in knowledge base")
                logger.info(f"  Query latency: {metrics['latency_ms']}ms (no results)")
                
        except Exception as e:
            logger.error(f"❌ Error in RAG lookup: {e}", exc_info=True)
            # Don't fail the entire turn if RAG fails, just log and continue
    
    async def correlate_speech_with_rag(self, speech_id: str, speech_timestamp: float) -> None:
        """
        Correlate a newly created speech with the most recent pending RAG event.
        
        This method is called when a speech is created to assign the speech ID
        to the RAG lookup that preceded it.
        
        Args:
            speech_id: The ID of the speech that was just created
            speech_timestamp: The timestamp when the speech was created
        """
        if not self.retriever.pending_rag_events:
            logger.debug("No pending RAG events to correlate")
            return
        
        # Find the most recent RAG event (should be the one that triggered this speech)
        # We take the last one added since RAG happens just before speech generation
        rag_event = self.retriever.pending_rag_events.pop(0)
        
        logger.info(f"Correlating speech {speech_id} with RAG event from {speech_timestamp - rag_event['timestamp']:.3f}s ago")
        
        # Send updated event with speech ID
        if self.event_callback:
            try:
                event_data = {**rag_event, "speechId": speech_id}
                # Remove timestamp from the event data sent to API
                event_data.pop("timestamp", None)
                
                await self.event_callback(
                    "knowledge_retrieved_with_speech",
                    event_data
                )
                logger.info(f"✅ Sent knowledge_retrieved_with_speech event for speech {speech_id}")
            except Exception as e:
                logger.error(f"Error sending knowledge_retrieved_with_speech event: {e}")


def create_knowledge_base_tool(retriever: KnowledgeBaseRetriever) -> Callable:
    """
    Create a function tool for querying knowledge bases.
    
    Args:
        retriever: KnowledgeBaseRetriever instance configured for this agent
        
    Returns:
        Function tool that can be added to agent's tools list
    """
    logger.info("🔧 Creating knowledge base tool function")
    logger.info(f"  Tool will query {len(retriever.knowledge_base_ids)} knowledge base(s)")
    
    @function_tool(description="Search the knowledge base for relevant information based on a query. Use this when you need to find specific information to answer the user's question.")
    async def query_knowledge(query: str) -> str:
        """
        Query the knowledge base for relevant information.
        
        Args:
            query: The search query to find relevant information
            
        Returns:
            Relevant context from the knowledge base
        """
        logger.info("🔧 Knowledge base tool called by LLM")
        logger.info(f"  Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        logger.info(f"  Query length: {len(query)} characters")
        
        context, metrics = await retriever.query_knowledge_base(query)
        
        # Send knowledge_retrieved event
        if retriever.event_callback and context:
            try:
                logger.info("  Sending knowledge_retrieved event to API...")
                await retriever.event_callback("knowledge_retrieved", {
                    "query": query,
                    "retrieved_context": context,
                    "context_length": len(context),
                    **metrics
                })
                logger.info("  ✓ Event sent successfully")
            except Exception as e:
                logger.error(f"  ❌ Error sending knowledge_retrieved event: {e}")
        
        if context:
            logger.info(f"✅ Knowledge base tool returned {len(context)} chars in {metrics['latency_ms']}ms")
            return context
        else:
            logger.warning("⚠️  Knowledge base tool found no relevant information")
            return "No relevant information found in the knowledge base."
    
    logger.info("✅ Knowledge base tool function created")
    return query_knowledge


def create_agent_from_config(
    agent_data: AgentData,
    tool_functions: Optional[List[Callable]] = None,
    event_callback: Optional[Callable] = None,
) -> Agent:
    """
    Create a LiveKit Agent from API agent configuration.
    
    This function converts agent configuration from the Voiceify API format
    into a LiveKit Agent instance with the appropriate instructions and tools.
    If the agent has knowledge bases assigned, creates a RAG-enabled agent.
    
    Tools are passed directly to the Agent constructor via the `tools` parameter.
    Each tool should already be decorated with @function_tool.
    
    Args:
        agent_data: Agent configuration from the API
        tool_functions: Optional list of tool functions (decorated with @function_tool)
        event_callback: Optional callback for sending events to API
        
    Returns:
        Configured Agent instance (or RAGAgent if knowledge bases are assigned)
        
    Example:
        >>> agent_data = AgentData(
        ...     id="agent-123",
        ...     name="Customer Support",
        ...     configuration={"instructions": "Help customers with issues"},
        ...     knowledge_base_ids=["kb-1", "kb-2"]
        ... )
        >>> agent = create_agent_from_config(agent_data)
    """
    instructions = agent_data.instructions
    
    logger.info(f"Creating agent: {agent_data.name} ({agent_data.id})")
    logger.debug(f"  Instructions preview: {instructions[:100]}...")
    
    # Initialize tool_functions list if None
    if tool_functions is None:
        tool_functions = []
    
    # Create appropriate agent type based on knowledge base configuration
    if agent_data.knowledge_base_ids:
        logger.info(f"  Agent has {len(agent_data.knowledge_base_ids)} knowledge base(s) assigned")
        
        if not agent_data.organization_id:
            logger.error(f"Agent {agent_data.name} has knowledge bases but no organization_id provided")
            raise ValueError(f"organization_id is required for knowledge base-enabled agents")
        
        # Check knowledge base configuration
        kb_config = agent_data.configuration.get("knowledgeBase", {})
        use_as_tool = kb_config.get("useAsTool", False)
        
        if use_as_tool:
            # Tool mode: add query_knowledge to tool functions
            logger.info(f"  Knowledge base mode: TOOL (LLM decides when to query)")
            logger.info("  Setting up knowledge base tool...")
            
            # Initialize Supabase and create retriever for tool mode
            logger.info("    Initializing Supabase client...")
            supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            voyage_api_key = os.getenv("VOYAGE_API_KEY")
            
            if not supabase_url or not supabase_key:
                raise ValueError("NEXT_PUBLIC_SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for knowledge base tools")
            if not voyage_api_key:
                raise ValueError("VOYAGE_API_KEY is required for knowledge base tools")
            if not SUPABASE_AVAILABLE:
                raise ImportError("Supabase SDK is not installed. Install with: pip install supabase")
            
            supabase_client: Client = create_client(supabase_url, supabase_key)
            logger.info("    ✓ Supabase client created")
            logger.info("    ✓ Voyage AI configured")
            
            # Create retriever
            logger.info("    Creating KnowledgeBaseRetriever for tool mode...")
            retriever = KnowledgeBaseRetriever(
                supabase_client=supabase_client,
                voyage_api_key=voyage_api_key,
                knowledge_base_ids=agent_data.knowledge_base_ids,
                event_callback=event_callback,
            )
            
            # Create and add knowledge base tool
            kb_tool = create_knowledge_base_tool(retriever)
            tool_functions.append(kb_tool)
            
            logger.info(f"  ✓ Added query_knowledge tool to agent")
            logger.info(f"  Total tools: {len(tool_functions)}")
            
            # Create standard Agent with KB tool
            logger.info("  Creating standard Agent with KB tool...")
            agent = Agent(instructions=instructions, tools=tool_functions)
            logger.info("  ✓ Agent created successfully")
        else:
            # Pre-injection mode: use RAGAgent (current behavior)
            logger.info(f"  Knowledge base mode: PRE-INJECT (automatic retrieval on user queries)")
            logger.info(f"  Total tools: {len(tool_functions)}")
            logger.info("  Creating RAGAgent with automatic knowledge retrieval...")
            
            agent = RAGAgent(
                agent_id=agent_data.id,
                organization_id=agent_data.organization_id,
                knowledge_base_ids=agent_data.knowledge_base_ids,
                instructions=instructions,
                tools=tool_functions,
                event_callback=event_callback,
            )
    else:
        # Create standard Agent without knowledge bases
        logger.info(f"  Creating standard agent (no knowledge bases)")
        logger.info(f"  Total tools: {len(tool_functions)}")
        
        if tool_functions:
            agent = Agent(instructions=instructions, tools=tool_functions)
            logger.info("  ✓ Standard agent with tools created")
        else:
            agent = Agent(instructions=instructions)
            logger.info("  ✓ Basic agent (no tools) created")
    
    # Store metadata for later use
    agent._voiceify_agent_id = agent_data.id
    agent._voiceify_agent_name = agent_data.name
    
    logger.info(f"Successfully created agent: {agent_data.name}")
    
    return agent

