"""
Voiceify LiveKit Voice Agent - New API Version

Main entry point using the new LiveKit Agents API with AgentSession.
Based on: https://docs.livekit.io/agents/quickstart/
"""

import logging
import asyncio
import os
from datetime import datetime
from typing import Optional
import aiohttp

from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.voice import (
    MetricsCollectedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    FunctionToolsExecutedEvent,
    UserInputTranscribedEvent,
    AgentStateChangedEvent,
    UserStateChangedEvent,
    SpeechCreatedEvent,
)
from livekit.plugins import openai, deepgram, elevenlabs, silero, noise_cancellation
from dotenv import load_dotenv

from .utils import Config, setup_logging
from .utils.telemetry import setup_langfuse
from .agents import AgentData, create_agent_from_config, AgentRegistry
from .tools import create_tool_functions
from .session import send_event_to_api, start_call_recording


# Set up logging
logger = setup_logging(level="INFO", name="voiceify-agent")

# Load environment variables at module level
load_dotenv("secrets.env")
load_dotenv(".env")


def prewarm(proc: JobProcess):
    """
    Prewarm function to load models before job starts.
    This runs once per worker process, not per job.
    """
    logger.info("Prewarming worker process...")
    # Load VAD model once for all jobs in this process
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("VAD model loaded")


async def entrypoint(ctx: JobContext):
    """
    Main entry point for voice agent jobs.
    
    This function is called for each new job (incoming call). It:
    1. Connects to the room and waits for a participant
    2. Fetches agent configuration from the API
    3. Sets up the voice pipeline (STT/LLM/TTS)
    4. Creates the agent with tools
    5. Starts the session
    6. Handles recording and events
    
    Args:
        ctx: Job context provided by LiveKit
    """
    # Load configuration
    try:
        config = Config.from_env()
        config.log_config(logger)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    
    # Add logging context
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    # Set up Langfuse telemetry with session metadata
    trace_provider = setup_langfuse(
        metadata={
            "langfuse.session.id": ctx.room.name,
        },
        host=config.langfuse_host,
        public_key=config.langfuse_public_key,
        secret_key=config.langfuse_secret_key,
    )
    
    # Add shutdown callback to flush traces
    if trace_provider:
        async def flush_trace():
            trace_provider.force_flush()
        
        ctx.add_shutdown_callback(flush_trace)
    
    logger.info("Connecting to room...")
    await ctx.connect()
    
    # Wait for participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant connected: {participant.identity}")
    
    # Extract call metadata from participant attributes
    phone_number = participant.attributes.get("sip.trunkPhoneNumber", "")
    twilio_call_sid = participant.attributes.get("sip.twilio.callSid", "")
    caller_phone_number = participant.attributes.get("sip.phoneNumber", "")
    called_phone_number = phone_number
    
    logger.info(f"Call context - Caller: {caller_phone_number}, Called: {called_phone_number}")
    
    # Generate session identifiers
    room_name = ctx.room.name or "unknown-room"
    session_id = f"session_{int(datetime.utcnow().timestamp() * 1000)}"
    session_start_time = datetime.utcnow()
    
    # Fetch agent configuration from API
    logger.info(f"Fetching agent configuration for phone number: {phone_number}")
    agent_data = await fetch_agent_config(config.api_url, phone_number)
    
    if not agent_data:
        logger.error("Failed to fetch agent configuration")
        return
    
    agent_id = agent_data.id
    organization_id = agent_data.organization_id or "unknown"
    agent_config = agent_data.configuration
    
    logger.info(f"Agent loaded: {agent_data.name} ({agent_id})")
    
    # Send room connected event
    await send_event_to_api(
        agent_id,
        twilio_call_sid,
        caller_phone_number,
        room_name,
        session_id,
        "room_connected",
        {"roomName": room_name},
        config.api_url,
    )
    
    # Create agent registry for transfers
    agent_registry = AgentRegistry()
    
    # Pre-instantiate related agents for transfers
    related_agents = agent_data.related_agents or {}
    logger.info(f"Pre-instantiating {len(related_agents)} related agents for transfers")
    
    for related_agent_id, related_agent_data in related_agents.items():
        logger.info(f"  - {related_agent_data.name} ({related_agent_id})")
        
        # Convert tools for related agent
        related_tools = related_agent_data.tools or []
        related_tool_functions = create_tool_functions(
            related_tools,
            agent_registry,
            caller_phone_number,
            called_phone_number,
            config.api_url,
            config.livekit_url,
            config.livekit_api_key,
            config.livekit_api_secret,
        )
        
        # Create event callback for this related agent
        async def send_related_agent_event(event_type: str, event_data: dict, aid=related_agent_id):
            """Send related agent-specific events to the API."""
            await send_event_to_api(
                aid,
                twilio_call_sid,
                caller_phone_number,
                room_name,
                session_id,
                event_type,
                event_data,
                config.api_url,
            )
        
        # Create and register the related agent
        related_agent = create_agent_from_config(
            related_agent_data,
            related_tool_functions,
            api_url=config.api_url,
            event_callback=send_related_agent_event,
        )
        agent_registry.register(related_agent_id, related_agent)
    
    # Convert tools for primary agent
    api_tools = agent_data.tools or []
    logger.info(f"Converting {len(api_tools)} tools for primary agent")
    
    primary_tool_functions = create_tool_functions(
        api_tools,
        agent_registry,
        caller_phone_number,
        called_phone_number,
        config.api_url,
        config.livekit_url,
        config.livekit_api_key,
        config.livekit_api_secret,
    )
    
    # Configure pipeline components (STT/LLM/TTS)
    logger.info("Configuring voice pipeline")
    
    pipeline_config = agent_config.get("pipeline", {})
    
    # Configure STT (Speech-to-Text)
    stt_config = pipeline_config.get("stt", {})
    stt_model_string = stt_config.get("model", "deepgram/nova-2-phonecall")
    stt_language = stt_config.get("language", "en")
    
    logger.info(f"STT: {stt_model_string} (language: {stt_language})")
    stt_provider, stt_model = stt_model_string.split("/") if "/" in stt_model_string else ("deepgram", stt_model_string)
    
    # Configure LLM
    llm_config = pipeline_config.get("llm", {})
    llm_model_string = llm_config.get("model", "openai/gpt-4o-mini")
    llm_temperature = llm_config.get("temperature", 0.8)
    
    logger.info(f"LLM: {llm_model_string} (temperature: {llm_temperature})")
    llm_provider, llm_model = llm_model_string.split("/") if "/" in llm_model_string else ("openai", llm_model_string)
    
    # Configure TTS (Text-to-Speech)
    tts_config = pipeline_config.get("tts", {})
    tts_model_string = tts_config.get("model", "elevenlabs/eleven_flash_v2_5:EXAVITQu4vr4xnSDxMaL")
    
    # Parse TTS model string: "provider/model:voice"
    if ":" in tts_model_string:
        tts_model_part, tts_voice = tts_model_string.split(":", 1)
    else:
        tts_model_part = tts_model_string
        tts_voice = "EXAVITQu4vr4xnSDxMaL"  # Default ElevenLabs voice
    
    tts_provider, tts_model = tts_model_part.split("/") if "/" in tts_model_part else ("elevenlabs", tts_model_part)
    
    logger.info(f"TTS: {tts_provider}/{tts_model} (voice: {tts_voice})")
    
    # Create the agent session using the new API
    logger.info("Creating agent session")
    session = AgentSession(
        stt=f"{stt_provider}/{stt_model}:{stt_language}",
        llm=f"{llm_provider}/{llm_model}",
        tts=f"{tts_provider}/{tts_model}:{tts_voice}",
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )
    
    # Set up metrics collection and event listeners
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """Handle metrics collection events."""
        try:
            # Log metrics using LiveKit's helper function
            metrics.log_metrics(ev.metrics)
            
            # Collect metrics for usage summary
            usage_collector.collect(ev.metrics)
            
            # Convert metrics to serializable format
            # Get all attributes from the metrics object
            metric = ev.metrics
            metrics_data = {
                "metric_type": type(metric).__name__,
            }
            
            # Serialize all attributes from the metrics object
            for attr_name in dir(metric):
                # Skip private/magic methods and callable attributes
                if not attr_name.startswith('_') and not callable(getattr(metric, attr_name, None)):
                    try:
                        value = getattr(metric, attr_name)
                        # Only include JSON-serializable types
                        if isinstance(value, (str, int, float, bool, type(None))):
                            metrics_data[attr_name] = value
                    except Exception:
                        # Skip attributes that can't be accessed
                        pass
            
            # Send to API
            asyncio.create_task(
                send_event_to_api(
                    agent_id,
                    twilio_call_sid,
                    caller_phone_number,
                    room_name,
                    session_id,
                    "metrics_collected",
                    metrics_data,
                    config.api_url,
                )
            )
        except Exception as e:
            logger.error(f"Error handling metrics: {e}")
    
    @session.on("conversation_item_added")
    def _on_conversation_item_added(ev: ConversationItemAddedEvent):
        """Handle conversation item added events."""
        try:
            # Serialize the conversation item
            item_data = {
                "type": ev.type,
                "created_at": ev.created_at,
            }
            
            # Serialize the item (ChatMessage or other types)
            if hasattr(ev.item, "model_dump"):
                item_data["item"] = ev.item.model_dump()
            else:
                for attr in ["type", "role", "text_content", "name", "call_id", "args", "output", "is_error"]:
                    if hasattr(ev.item, attr):
                        value = getattr(ev.item, attr)
                        if isinstance(value, (str, int, float, bool, type(None), dict, list)):
                            item_data[attr] = value
            
            asyncio.create_task(
                send_event_to_api(
                    agent_id,
                    twilio_call_sid,
                    caller_phone_number,
                    room_name,
                    session_id,
                    "conversation_item_added",
                    item_data,
                    config.api_url,
                )
            )
        except Exception as e:
            logger.error(f"Error handling conversation_item_added: {e}")
    
    @session.on("function_tools_executed")
    def _on_function_tools_executed(ev: FunctionToolsExecutedEvent):
        """Handle function tool execution events."""
        try:
            # Serialize function tool execution data
            tools_data = {
                "type": ev.type,
                "created_at": ev.created_at,
                "function_calls": [call.model_dump() if hasattr(call, "model_dump") else str(call) for call in ev.function_calls],
                "function_call_outputs": [
                    output.model_dump() if output and hasattr(output, "model_dump") else str(output) 
                    for output in ev.function_call_outputs
                ],
                "has_tool_reply": ev.has_tool_reply,
                "has_agent_handoff": ev.has_agent_handoff,
            }
            
            asyncio.create_task(
                send_event_to_api(
                    agent_id,
                    twilio_call_sid,
                    caller_phone_number,
                    room_name,
                    session_id,
                    "function_tools_executed",
                    tools_data,
                    config.api_url,
                )
            )
        except Exception as e:
            logger.error(f"Error handling function_tools_executed: {e}")
    
    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: UserInputTranscribedEvent):
        """Handle user input transcription events."""
        try:
            # Only send final transcriptions
            if ev.is_final:
                transcription_data = {
                    "type": ev.type,
                    "transcript": ev.transcript,
                    "is_final": ev.is_final,
                    "created_at": ev.created_at,
                }
                
                # Add optional fields if present
                if ev.speaker_id:
                    transcription_data["speaker_id"] = ev.speaker_id
                if ev.language:
                    transcription_data["language"] = ev.language
                
                asyncio.create_task(
                    send_event_to_api(
                        agent_id,
                        twilio_call_sid,
                        caller_phone_number,
                        room_name,
                        session_id,
                        "user_input_transcribed",
                        transcription_data,
                        config.api_url,
                    )
                )
        except Exception as e:
            logger.error(f"Error handling user_input_transcribed: {e}")
    
    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: AgentStateChangedEvent):
        """Handle agent state change events."""
        try:
            state_data = {
                "type": ev.type,
                "old_state": ev.old_state,
                "new_state": ev.new_state,
                "created_at": ev.created_at,
            }
            
            asyncio.create_task(
                send_event_to_api(
                    agent_id,
                    twilio_call_sid,
                    caller_phone_number,
                    room_name,
                    session_id,
                    "agent_state_changed",
                    state_data,
                    config.api_url,
                )
            )
        except Exception as e:
            logger.error(f"Error handling agent_state_changed: {e}")
    
    @session.on("user_state_changed")
    def _on_user_state_changed(ev: UserStateChangedEvent):
        """Handle user state change events."""
        try:
            state_data = {
                "type": ev.type,
                "old_state": ev.old_state,
                "new_state": ev.new_state,
                "created_at": ev.created_at,
            }
            
            asyncio.create_task(
                send_event_to_api(
                    agent_id,
                    twilio_call_sid,
                    caller_phone_number,
                    room_name,
                    session_id,
                    "user_state_changed",
                    state_data,
                    config.api_url,
                )
            )
        except Exception as e:
            logger.error(f"Error handling user_state_changed: {e}")
    
    @session.on("speech_created")
    def _on_speech_created(ev: SpeechCreatedEvent):
        """Handle speech creation events."""
        try:
            speech_data = {
                "type": ev.type,
                "user_initiated": ev.user_initiated,
                "source": ev.source,
                "created_at": ev.created_at,
            }
            
            # Note: speech_handle is excluded from serialization in the event model
            # but we can try to extract text from it if available
            if hasattr(ev.speech_handle, "text"):
                speech_data["text"] = ev.speech_handle.text
            
            asyncio.create_task(
                send_event_to_api(
                    agent_id,
                    twilio_call_sid,
                    caller_phone_number,
                    room_name,
                    session_id,
                    "speech_created",
                    speech_data,
                    config.api_url,
                )
            )
        except Exception as e:
            logger.error(f"Error handling speech_created: {e}")
    
    # Track session data for final summary
    session_data = {
        "startedAt": session_start_time.isoformat() + "Z",
        "config": agent_config,
        "metadata": dict(participant.attributes),
    }
    
    # Track recording info (will be populated when recording is enabled)
    recording_info = {"egress_id": None, "recording_filename": None}
    
    # Add session close handler to send complete session data with usage summary
    @session.on("close")
    def _on_session_close(ev: CloseEvent):
        """Handle session close event and send complete session data."""
        try:
            # Log close reason and any errors
            logger.info(f"Session closing - reason: {ev.reason}")
            if ev.error:
                logger.error(f"Session closed with error: {ev.error}")
            
            # Get usage summary
            summary = usage_collector.get_summary()
            
            logger.info("\n" + "=" * 80)
            logger.info("📊 SESSION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Usage summary: {summary}")
            logger.info("=" * 80 + "\n")
            
            # Calculate session duration
            session_end_time = datetime.utcnow()
            session_duration_ms = int((session_end_time - session_start_time).total_seconds() * 1000)
            
            # Construct recording URL if we have a recording
            recording_url = None
            if recording_info["recording_filename"]:
                supabase_project_ref = os.getenv("SUPABASE_PROJECT_REF")
                if supabase_project_ref:
                    recording_url = f"https://{supabase_project_ref}.supabase.co/storage/v1/object/public/call-recordings/{recording_info['recording_filename']}"
                    logger.info(f"📹 Recording URL: {recording_url}")
            
            # Send complete session data to API (including initial session data)
            # Note: Using create_task as this is a sync handler; shutdown callback also sends data
            
            # Convert UsageSummary to dict for JSON serialization (camelCase for API)
            usage_dict = {
                "llmPromptTokens": summary.llm_prompt_tokens,
                "llmPromptCachedTokens": summary.llm_prompt_cached_tokens,
                "llmInputAudioTokens": summary.llm_input_audio_tokens,
                "llmCompletionTokens": summary.llm_completion_tokens,
                "llmOutputAudioTokens": summary.llm_output_audio_tokens,
                "ttsCharactersCount": summary.tts_characters_count,
                "ttsAudioDuration": summary.tts_audio_duration,
                "sttAudioDuration": summary.stt_audio_duration,
            }
            
            asyncio.create_task(
                send_event_to_api(
                    agent_id,
                    twilio_call_sid,
                    caller_phone_number,
                    room_name,
                    session_id,
                    "session_complete",
                    {
                        **session_data,  # Include startedAt, config, metadata
                        "endedAt": session_end_time.isoformat() + "Z",
                        "durationMs": session_duration_ms,
                        "egressId": recording_info["egress_id"],
                        "recordingUrl": recording_url,
                        "usage": usage_dict,
                    },
                    config.api_url,
                )
            )
            
            logger.info("✅ Session complete event sent")
            
        except Exception as e:
            logger.error(f"❌ Error handling session close: {e}")
    
    # Register shutdown callback for usage summary
    async def log_usage():
        """Log usage summary from collected metrics."""
        try:
            summary = usage_collector.get_summary()
            logger.info(f"Session usage summary: {summary}")
        except Exception as e:
            logger.error(f"Error generating usage summary: {e}")
    
    ctx.add_shutdown_callback(log_usage)
    
    # Register shutdown callback for sending final transcript
    async def send_transcript():
        """Send final transcript when session shuts down."""
        logger.info("🔚 Shutdown callback triggered - sending final transcript...")
        
        try:
            # Access the conversation history from the session
            history_json = session.history.to_dict(
                exclude_timestamp=False,
                exclude_image=True,
                exclude_audio=True,
            )
            
            # Convert history items to serializable format
            transcript_items = []
            for item in session.history.items:
                if item.type == "message":
                    transcript_items.append({
                        "type": "message",
                        "role": item.role,
                        "content": item.text_content or "",
                        "timestamp": item.created_at,
                    })
                elif item.type == "function_call":
                    # Use model_dump if available, otherwise construct manually
                    if hasattr(item, "model_dump"):
                        transcript_items.append({
                            "type": "function_call",
                            **item.model_dump(),
                            "timestamp": item.created_at,
                        })
                    else:
                        # Safely access attributes that may exist
                        function_call_data = {
                            "type": "function_call",
                            "timestamp": item.created_at,
                        }
                        # Add attributes that exist
                        for attr in ["name", "call_id", "arguments"]:
                            if hasattr(item, attr):
                                value = getattr(item, attr)
                                # Convert to camelCase for API
                                key = "callId" if attr == "call_id" else "args" if attr == "arguments" else attr
                                function_call_data[key] = value
                        transcript_items.append(function_call_data)
                elif item.type == "function_call_output":
                    # Use model_dump if available, otherwise construct manually
                    if hasattr(item, "model_dump"):
                        transcript_items.append({
                            "type": "function_call_output",
                            **item.model_dump(),
                            "timestamp": item.created_at,
                        })
                    else:
                        # Safely access attributes that may exist
                        output_data = {
                            "type": "function_call_output",
                            "timestamp": item.created_at,
                        }
                        # Add attributes that exist
                        for attr in ["name", "call_id", "output", "is_error"]:
                            if hasattr(item, attr):
                                value = getattr(item, attr)
                                # Convert to camelCase for API
                                key = "callId" if attr == "call_id" else "isError" if attr == "is_error" else attr
                                output_data[key] = value
                        transcript_items.append(output_data)
            
            # Send transcript to API
            await send_event_to_api(
                agent_id,
                twilio_call_sid,
                caller_phone_number,
                room_name,
                session_id,
                "transcript",
                {
                    "history": history_json,
                    "items": transcript_items,
                },
                config.api_url,
            )
            
            logger.info("✅ Transcript sent successfully")
            
        except Exception as e:
            logger.error(f"❌ Error sending transcript in shutdown callback: {e}")
    
    ctx.add_shutdown_callback(send_transcript)
    
    # Create event callback for RAG agents to send knowledge_retrieved events
    async def send_agent_event(event_type: str, event_data: dict):
        """Send agent-specific events to the API."""
        await send_event_to_api(
            agent_id,
            twilio_call_sid,
            caller_phone_number,
            room_name,
            session_id,
            event_type,
            event_data,
            config.api_url,
        )
    
    # Create the primary agent with tools
    primary_agent = create_agent_from_config(
        agent_data,
        primary_tool_functions,
        api_url=config.api_url,
        event_callback=send_agent_event,
    )
    
    # Start the session
    logger.info("Starting voice agent session")
    
    noise_cancellation_enabled = agent_config.get("noiseCancellation", {}).get("enabled", False)
    
    # Prepare room input options with noise cancellation
    # Using BVCTelephony for telephony applications as per docs
    # https://docs.livekit.io/home/cloud/noise-cancellation/
    await session.start(
        agent=primary_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVCTelephony() if noise_cancellation_enabled else None,
        )
    )
    
    if noise_cancellation_enabled:
        logger.info("Noise cancellation enabled (BVCTelephony model)")
    
    logger.info("Voice agent session started successfully")
    
    # Start call recording if configured
    if agent_config.get("recording", {}).get("enabled", False):
        logger.info("Starting call recording...")
        egress_id, recording_filename = await start_call_recording(
            config.livekit_url,
            config.livekit_api_key,
            config.livekit_api_secret,
            organization_id,
            session_id,
            room_name,
            config.supabase_project_ref,
            config.supabase_s3_access_key,
            config.supabase_s3_secret_key,
            config.supabase_s3_region,
        )
        
        if egress_id and recording_filename:
            recording_info["egress_id"] = egress_id
            recording_info["recording_filename"] = recording_filename
            logger.info(f"🎬 Started call recording with egress ID: {egress_id}")
        else:
            logger.error("❌ Failed to start call recording")
    
    # Handle first message
    settings = agent_config.get("settings", {})
    if settings.get("generateFirstMessage", False):
        first_message_type = settings.get("firstMessageType", "generated")
        
        if first_message_type == "direct":
            # Use direct message
            first_message = settings.get("firstMessage", "Hello. How can I help you today?")
            logger.info(f"Saying direct first message: {first_message}")
            session.say(first_message)
        else:
            # Use AI-generated message
            first_message_prompt = settings.get("firstMessagePrompt", "Greet the user.")
            logger.info(f"Generating first message with prompt: {first_message_prompt}")
            session.generate_reply(instructions=first_message_prompt)


async def fetch_agent_config(api_url: str, phone_number: str) -> Optional[AgentData]:
    """
    Fetch agent configuration from the API.
    
    Args:
        api_url: Base URL for the API
        phone_number: Phone number to look up agent configuration
        
    Returns:
        AgentData instance or None if fetch fails
    """
    try:
        endpoint = f"{api_url}/api/phone-number/{phone_number}/agent"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(endpoint) as response:
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"Failed to fetch agent config: {response.status} {error_text}")
                    return None
                
                data = await response.json()
                
                # Convert to AgentData
                return AgentData(
                    id=data.get("id"),
                    name=data.get("name", "Unknown Agent"),
                    configuration=data.get("configuration", {}),
                    tools=data.get("tools", []),
                    organization_id=data.get("organization_id"),
                    related_agents={
                        agent_id: AgentData(
                            id=agent_id,
                            name=agent_info.get("name", "Unknown"),
                            configuration=agent_info.get("configuration", {}),
                            tools=agent_info.get("tools", []),
                        )
                        for agent_id, agent_info in data.get("relatedAgents", {}).items()
                    },
                    has_knowledge_base=data.get("hasKnowledgeBase", False),
                )
                
    except Exception as e:
        logger.error(f"Error fetching agent configuration: {e}")
        return None


if __name__ == "__main__":
    """Run the worker."""
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )

