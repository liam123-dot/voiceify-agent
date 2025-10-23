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
    inference,
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
from livekit.plugins import openai, deepgram, elevenlabs, silero, noise_cancellation, groq
from livekit.plugins.turn_detector.english import EnglishModel
from dotenv import load_dotenv

from .utils import Config, setup_logging
from .utils.telemetry import setup_langfuse
from .agents import AgentData, create_agent_from_config, AgentRegistry
from .tools import create_tool_functions
from .session import send_event_to_api, start_call_recording


# Set up logging
logger = setup_logging(level="INFO", name="voiceify-agent")

# Load environment variables at module level
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
    stt_model_string = stt_config.get("model", "deepgram/nova-3")
    stt_language = stt_config.get("language", "en")
    stt_inference_type = stt_config.get("inferenceType", "livekit")

    logger.info(f"STT: {stt_model_string} (language: {stt_language}, inference: {stt_inference_type})")
    stt_provider, stt_model = stt_model_string.split("/") if "/" in stt_model_string else ("deepgram", stt_model_string)

    # Create STT instance based on inference type
    if stt_inference_type == "livekit":
        # Use LiveKit Inference API for managed STT
        stt_instance = inference.STT(
            model=stt_model_string,  # Full model string with provider
            language=stt_language,
        )
        logger.info(f"Created LiveKit Inference STT with model={stt_model_string}, language={stt_language}")
    else:
        # Use direct provider plugin
        if stt_provider == "deepgram":
            stt_instance = deepgram.STT(
                model=stt_model,
                language=stt_language,
            )
            logger.info(f"Created Deepgram STT plugin with model={stt_model}, language={stt_language}")
        else:
            raise ValueError(f"Unsupported STT provider for direct inference: {stt_provider}")
    
    # Configure LLM
    llm_config = pipeline_config.get("llm", {})
    llm_model_string = llm_config.get("model", "openai/gpt-4o-mini")
    llm_temperature = llm_config.get("temperature", 0.8)
    llm_inference_type = llm_config.get("inferenceType", "livekit")

    logger.info(f"LLM: {llm_model_string} (temperature: {llm_temperature}, inference: {llm_inference_type})")
    if "/" in llm_model_string:
        llm_provider, llm_model = llm_model_string.split("/", 1)
    else:
        llm_provider, llm_model = "openai", llm_model_string

    # Create LLM instance based on inference type
    if llm_inference_type == "livekit":
        # Use LiveKit Inference API for managed model hosting
        llm_instance = inference.LLM(
            model=llm_model_string,  # Full model string with provider
            provider=llm_provider,
            extra_kwargs={
                "temperature": llm_temperature,
            }
        )
        logger.info(f"Created LiveKit Inference LLM with model={llm_model_string}, temperature={llm_temperature}")
    else:
        # Use direct provider plugin
        if llm_provider == "openai":
            llm_instance = openai.LLM(
                model=llm_model,
                temperature=llm_temperature,
            )
            logger.info(f"Created OpenAI LLM instance with model={llm_model}, temperature={llm_temperature}")
        elif llm_provider == "groq":
            llm_instance = groq.LLM(
                model=llm_model,
                temperature=llm_temperature,
            )
            logger.info(f"Created Groq LLM instance with model={llm_model}, temperature={llm_temperature}")
        else:
            raise ValueError(f"Unsupported LLM provider for direct inference: {llm_provider}")
    
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
    
    # Create TTS plugin instance
    if tts_provider == "elevenlabs":
        # Extract voice settings from TTS config (with ElevenLabs recommended defaults)
        tts_stability = tts_config.get("stability", 0.5)
        tts_similarity = tts_config.get("similarity_boost", 0.75)
        tts_style = tts_config.get("style", 0.0)
        tts_speaker_boost = tts_config.get("use_speaker_boost", True)
        tts_speed = tts_config.get("speed", 1.0)

        tts_instance = elevenlabs.TTS(
            model=tts_model,
            voice_id=tts_voice,
            api_key=config.elevenlabs_api_key,
            voice_settings=elevenlabs.VoiceSettings(
                stability=tts_stability,
                similarity_boost=tts_similarity,
                style=tts_style,
                use_speaker_boost=tts_speaker_boost,
                speed=tts_speed,
            ),
            chunk_length_schedule=[50, 80, 120, 160]
        )
        logger.info(
            f"Created ElevenLabs TTS instance with model={tts_model}, voice={tts_voice}, "
            f"stability={tts_stability}, similarity_boost={tts_similarity}, style={tts_style}, "
            f"speaker_boost={tts_speaker_boost}, speed={tts_speed}"
        )
    else:
        raise ValueError(f"Unsupported TTS provider: {tts_provider}")
    
    # Configure turn detection and VAD
    turn_detection_config = agent_config.get("turnDetection", {})
    turn_detection_type = turn_detection_config.get("type", "multilingual")
    vad_options = turn_detection_config.get("vadOptions", {})
    turn_detector_options = turn_detection_config.get("turnDetectorOptions", {})
    
    logger.info(f"Turn detection type: {turn_detection_type}")
    logger.info(f"VAD options: {vad_options}")
    
    # Create VAD instance with custom options (convert ms to seconds)
    vad_instance = silero.VAD.load(
        min_speech_duration=vad_options.get("minSpeechDuration", 50) / 1000.0,
        min_silence_duration=vad_options.get("minSilenceDuration", 550) / 1000.0,
        prefix_padding_duration=vad_options.get("prefixPaddingDuration", 500) / 1000.0,
    )
    logger.info(f"Created VAD with min_speech={vad_options.get('minSpeechDuration', 50)}ms, "
                f"min_silence={vad_options.get('minSilenceDuration', 550)}ms, "
                f"prefix_padding={vad_options.get('prefixPaddingDuration', 500)}ms")
    
    # Create turn detection instance based on type
    # Note: EnglishModel itself has no configuration parameters
    turn_detection_instance = None
    if turn_detection_type == "multilingual":
        turn_detection_instance = EnglishModel()
        logger.info("Created EnglishModel turn detector for semantic end-of-turn detection")
    elif turn_detection_type == "server-vad":
        logger.info("Using server VAD only (no semantic turn detection)")
        turn_detection_instance = None
    else:
        logger.info("Turn detection disabled")
        turn_detection_instance = None
    
    # Extract endpointing delay parameters for AgentSession (convert ms to seconds)
    min_endpointing_delay = turn_detector_options.get("minEndpointingDelay", 500) / 1000.0
    max_endpointing_delay = turn_detector_options.get("maxEndpointingDelay", 6000) / 1000.0
    
    logger.info(f"Endpointing delays: min={min_endpointing_delay}s, max={max_endpointing_delay}s")
    
    # Create the agent session using the new API with configured VAD and turn detection
    logger.info("Creating agent session with configured VAD and turn detection")
    session = AgentSession(
        stt=stt_instance,
        llm=llm_instance,
        tts=tts_instance,
        vad=vad_instance,
        turn_detection=turn_detection_instance,
        min_endpointing_delay=min_endpointing_delay,
        max_endpointing_delay=max_endpointing_delay,
        
        # preemptive_generation=True,
    )
    
    # Set up metrics collection and event listeners
    usage_collector = metrics.UsageCollector()
    
    # Track metrics by speech_id for latency calculation
    speech_metrics: dict[str, dict] = {}
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        """Handle metrics collection events."""
        try:
            # Log metrics using LiveKit's helper function
            metrics.log_metrics(ev.metrics)
            
            # Collect metrics for usage summary
            usage_collector.collect(ev.metrics)
            
            # Extract structured metrics based on type
            metric = ev.metrics
            metrics_data = None
            speech_id = None
            
            # End-of-Utterance Metrics
            if isinstance(metric, metrics.EOUMetrics):
                speech_id = metric.speech_id
                metrics_data = {
                    "metricType": "eou",
                    "endOfUtteranceDelay": metric.end_of_utterance_delay,
                    "transcriptionDelay": metric.transcription_delay,
                    "onUserTurnCompletedDelay": metric.on_user_turn_completed_delay,
                    "speechId": speech_id,
                }
                
                # Store for latency calculation
                # Use the minimum of the three delays as it's the most accurate indicator
                # of when the user actually stopped speaking
                eou_delay = min(
                    metric.end_of_utterance_delay,
                    metric.transcription_delay,
                    metric.on_user_turn_completed_delay
                )
                
                if speech_id:
                    if speech_id not in speech_metrics:
                        speech_metrics[speech_id] = {}
                    speech_metrics[speech_id]["eou"] = eou_delay
                    
                logger.info(
                    f"📊 EOU Metrics - Speech ID: {speech_id}, "
                    f"EOU: {metric.end_of_utterance_delay:.3f}s, "
                    f"Transcription: {metric.transcription_delay:.3f}s, "
                    f"TurnCompleted: {metric.on_user_turn_completed_delay:.3f}s, "
                    f"Using: {eou_delay:.3f}s"
                )
            
            # Speech-to-Text Metrics
            elif isinstance(metric, metrics.STTMetrics):
                metrics_data = {
                    "metricType": "stt",
                    "audioDuration": metric.audio_duration,
                    "duration": metric.duration,
                    "streamed": metric.streamed,
                }
                
                logger.info(f"📊 STT Metrics - Audio: {metric.audio_duration:.3f}s, Duration: {metric.duration:.3f}s")
            
            # Large Language Model Metrics
            elif isinstance(metric, metrics.LLMMetrics):
                speech_id = metric.speech_id
                metrics_data = {
                    "metricType": "llm",
                    "duration": metric.duration,
                    "completionTokens": metric.completion_tokens,
                    "promptTokens": metric.prompt_tokens,
                    "promptCachedTokens": metric.prompt_cached_tokens,
                    "ttft": metric.ttft,
                    "tokensPerSecond": metric.tokens_per_second,
                    "speechId": speech_id,
                    "totalTokens": metric.total_tokens,
                }
                
                # Store for latency calculation
                if speech_id:
                    if speech_id not in speech_metrics:
                        speech_metrics[speech_id] = {}
                    speech_metrics[speech_id]["llm_ttft"] = metric.ttft
                    
                logger.info(f"📊 LLM Metrics - Speech ID: {speech_id}, TTFT: {metric.ttft:.3f}s, Tokens/s: {metric.tokens_per_second:.1f}")
            
            # Text-to-Speech Metrics
            elif isinstance(metric, metrics.TTSMetrics):
                speech_id = metric.speech_id
                metrics_data = {
                    "metricType": "tts",
                    "audioDuration": metric.audio_duration,
                    "charactersCount": metric.characters_count,
                    "duration": metric.duration,
                    "ttfb": metric.ttfb,
                    "speechId": speech_id,
                    "streamed": metric.streamed,
                }
                
                # Store for latency calculation
                if speech_id:
                    if speech_id not in speech_metrics:
                        speech_metrics[speech_id] = {}
                    speech_metrics[speech_id]["tts_ttfb"] = metric.ttfb
                    
                logger.info(f"📊 TTS Metrics - Speech ID: {speech_id}, TTFB: {metric.ttfb:.3f}s, Chars: {metric.characters_count}")
            
            # Voice Activity Detection Metrics
            elif isinstance(metric, metrics.VADMetrics):
                metrics_data = {
                    "metricType": "vad",
                    "idleTime": metric.idle_time,
                    "inferenceCount": metric.inference_count,
                    "inferenceDurationTotal": metric.inference_duration_total,
                    "label": metric.label if hasattr(metric, "label") else None,
                }
                
                logger.info(f"📊 VAD Metrics - Inferences: {metric.inference_count}, Idle: {metric.idle_time:.3f}s")
            
            # Unknown metric type - fallback to generic serialization
            else:
                metrics_data = {
                    "metricType": "unknown",
                    "typeName": type(metric).__name__,
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
                
                logger.warning(f"⚠️  Unknown metric type: {type(metric).__name__}")
            
            # Send structured metrics to API
            if metrics_data:
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
            
            # Calculate total latency if we have all three components for this speech_id
            if speech_id and speech_id in speech_metrics:
                turn_metrics = speech_metrics[speech_id]
                if all(k in turn_metrics for k in ["eou", "llm_ttft", "tts_ttfb"]):
                    # RAG latency will be added separately via knowledge_retrieved_with_speech events
                    total_latency = turn_metrics["eou"] + turn_metrics["llm_ttft"] + turn_metrics["tts_ttfb"]
                    
                    latency_data = {
                        "metricType": "total_latency",
                        "speechId": speech_id,
                        "totalLatency": total_latency,
                        "eouDelay": turn_metrics["eou"],
                        "llmTtft": turn_metrics["llm_ttft"],
                        "ttsTtfb": turn_metrics["tts_ttfb"],
                    }
                    
                    logger.info(
                        f"📊 Total Latency - Speech ID: {speech_id}, "
                        f"Total: {total_latency:.3f}s "
                        f"(EOU: {turn_metrics['eou']:.3f}s + "
                        f"LLM: {turn_metrics['llm_ttft']:.3f}s + "
                        f"TTS: {turn_metrics['tts_ttfb']:.3f}s) "
                        f"[RAG latency tracked separately]"
                    )
                    
                    # Send total latency event
                    asyncio.create_task(
                        send_event_to_api(
                            agent_id,
                            twilio_call_sid,
                            caller_phone_number,
                            room_name,
                            session_id,
                            "total_latency",
                            latency_data,
                            config.api_url,
                        )
                    )
                    
                    # Clean up stored metrics for this speech_id
                    del speech_metrics[speech_id]
                    
        except Exception as e:
            logger.error(f"Error handling metrics: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
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
        """Handle speech creation events and capture TTS-aligned transcriptions."""
        try:
            speech_data = {
                "type": ev.type,
                "user_initiated": ev.user_initiated,
                "source": ev.source,
                "created_at": ev.created_at,
            }
            
            # Extract text and aligned transcription data from speech handle
            if hasattr(ev.speech_handle, "text"):
                speech_data["text"] = ev.speech_handle.text
            
            # Check for TTS-aligned transcription with timestamps
            # The speech_handle may have transcription data with word-level timing
            if hasattr(ev.speech_handle, "transcription"):
                transcription = ev.speech_handle.transcription
                speech_data["transcription"] = {
                    "text": str(transcription) if transcription else None,
                }
            
            # Extract speech ID for RAG correlation
            speech_id = None
            if hasattr(ev.speech_handle, "speech_id"):
                speech_id = ev.speech_handle.speech_id
                speech_data["speech_id"] = speech_id
            elif hasattr(ev.speech_handle, "id"):
                speech_id = ev.speech_handle.id
                speech_data["speech_id"] = speech_id
                
            # Check for word-level timing information
            if hasattr(ev.speech_handle, "words") and ev.speech_handle.words:
                speech_data["words"] = [
                    {
                        "word": word.text if hasattr(word, "text") else str(word),
                        "startTime": word.start_time if hasattr(word, "start_time") else None,
                        "endTime": word.end_time if hasattr(word, "end_time") else None,
                    }
                    for word in ev.speech_handle.words
                ]
                logger.info(f"📝 Captured {len(ev.speech_handle.words)} words with timestamps")
            
            # Correlate with pending RAG events if this is a RAGAgent
            if speech_id and hasattr(primary_agent, 'correlate_speech_with_rag'):
                import time
                asyncio.create_task(
                    primary_agent.correlate_speech_with_rag(speech_id, time.time())
                )
            
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
            if recording_info["recording_filename"] and config.supabase_project_ref:
                recording_url = f"https://{config.supabase_project_ref}.supabase.co/storage/v1/object/public/call-recordings/{recording_info['recording_filename']}"
                logger.info(f"📹 Recording URL: {recording_url}")
            elif recording_info["recording_filename"]:
                logger.warning("⚠️  Recording filename exists but Supabase project ref is missing")
            else:
                logger.info("ℹ️  No recording for this session")
            
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
        event_callback=send_agent_event,
    )
    
    # Start call recording BEFORE session starts (per LiveKit docs)
    # This ensures the entire session is captured from the beginning
    # Check if we have the necessary Supabase credentials for recording
    if (config.supabase_project_ref and config.supabase_s3_access_key and 
        config.supabase_s3_secret_key and config.supabase_s3_region):
        logger.info("Starting call recording...")
        egress_id, recording_filename, started_at_unix_ms = await start_call_recording(
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
        
        if egress_id and recording_filename and started_at_unix_ms:
            recording_info["egress_id"] = egress_id
            recording_info["recording_filename"] = recording_filename
            recording_info["started_at_unix_ms"] = started_at_unix_ms
            
            # Send recording_started event with precise timestamp from LiveKit
            # This timestamp represents when the recording actually began capturing audio,
            # which is critical for accurate timeline synchronization in the UI
            await send_event_to_api(
                agent_id,
                twilio_call_sid,
                caller_phone_number,
                room_name,
                session_id,
                "recording_started",
                {
                    "egressId": egress_id,
                    "startedAtUnixMs": started_at_unix_ms,
                    "recordingFilename": recording_filename,
                },
                config.api_url,
            )
            
            logger.info(f"🎬 Started call recording with egress ID: {egress_id}")
            logger.info(f"📁 Recording filename: {recording_filename}")
            logger.info(f"⏱️  Recording started at: {started_at_unix_ms}")
        else:
            logger.error("❌ Failed to start call recording")
    else:
        logger.warning("⚠️  Call recording disabled: Missing Supabase credentials")
        logger.debug(f"Supabase config - Project: {config.supabase_project_ref}, "
                    f"Access Key: {'SET' if config.supabase_s3_access_key else 'NOT SET'}, "
                    f"Secret Key: {'SET' if config.supabase_s3_secret_key else 'NOT SET'}, "
                    f"Region: {config.supabase_s3_region}")
    
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

    logger.info(f"audio output enabled: {session.output.audio_enabled}")
    logger.info(f"audio output: {session.output.audio}")
    
    if noise_cancellation_enabled:
        logger.info("Noise cancellation enabled (BVCTelephony model)")
    
    logger.info("Voice agent session started successfully")
    
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
                            organization_id=agent_info.get("organization_id"),
                            knowledge_base_ids=agent_info.get("knowledgeBaseIds", []),
                        )
                        for agent_id, agent_info in data.get("relatedAgents", {}).items()
                    },
                    knowledge_base_ids=data.get("knowledgeBaseIds", []),
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

