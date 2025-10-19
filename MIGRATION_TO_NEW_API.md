# Migration to New LiveKit Agents API

This document describes the migration from the old LiveKit Agents API to the new API.

## Overview

The codebase has been updated to use the new LiveKit Agents API which features:
- `Agent` class instead of `VoiceAssistant`
- `AgentSession` for managing sessions
- `@function_tool()` decorator for tools
- Agent handoffs via tool return values
- Improved telemetry with Langfuse OpenTelemetry integration

## Key Changes

### 1. Agent Creation (`src/agents/factory.py`)
- Changed from `VoiceAssistant` to `Agent` class
- Tools are now added as `@function_tool()` decorated methods on dynamic Agent subclasses
- Agents are created with instructions and tools together

### 2. Tool Conversion (`src/tools/converter_new.py`)
- New tool converter creates `@function_tool()` decorated functions
- Agent-to-agent transfers return Agent instances from tools (per [workflows docs](https://docs.livekit.io/agents/build/workflows/))
- Phone transfers use `TransferSIPParticipantRequest` API (per [SIP transfer docs](https://docs.livekit.io/sip/transfer-cold/))
- API tools execute via HTTP and raise `ToolError` on failure

### 3. Session Management (`src/agent_new.py`)
- Uses `AgentSession` instead of `voice.AgentSession`
- Session configuration uses string format: `"provider/model:voice"`
- Simplified session start with `agent` parameter
- Proper metrics collection with `@session.on("metrics_collected")`

### 4. Telemetry (`src/utils/telemetry.py`)
- Langfuse setup at module level (not per-job)
- OpenTelemetry integration for traces and metrics
- Usage collector for aggregating metrics per session

### 5. Prewarm Function
- Added `prewarm()` function to load VAD model once per worker process
- Improves performance by reducing cold start time

## Files Changed

### New Files
- `src/agent_new.py` - New entrypoint using latest API
- `src/tools/converter_new.py` - New tool converter for `@function_tool` decorators
- `src/utils/telemetry.py` - Langfuse OpenTelemetry setup

### Modified Files
- `src/agents/factory.py` - Updated to create `Agent` instances with tools
- `src/agents/registry.py` - Changed type from `VoiceAssistant` to `Agent`
- `src/utils/config.py` - Added Langfuse configuration
- `requirements.txt` - Added OpenTelemetry dependencies
- `secrets.env` - Changed `LANGFUSE_BASE_URL` to `LANGFUSE_HOST`
- `env.example` - Added Langfuse configuration template

## Migration Steps

### To use the new API:

1. **Install new dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Update environment variables:**
   - Rename `LANGFUSE_BASE_URL` to `LANGFUSE_HOST` in your `.env` file

3. **Use the new entrypoint:**
   ```bash
   # Instead of: python -m src.agent
   python -m src.agent_new
   ```

4. **Update Dockerfile (if needed):**
   ```dockerfile
   CMD ["python", "-m", "src.agent_new"]
   ```

## New API Benefits

### 1. Better Tool Integration
- Tools are first-class methods on Agent classes
- Type-safe tool definitions with decorators
- Automatic schema generation from function signatures

### 2. Agent Handoffs
- Clean handoff pattern via tool return values
- Context preservation via `chat_ctx` parameter
- Support for multi-agent workflows

### 3. Improved Observability
- Full OpenTelemetry tracing with Langfuse
- Automatic metrics collection for STT, LLM, and TTS
- Usage summaries for cost estimation

### 4. Simplified Configuration
- String-based model configuration
- Cleaner session initialization
- Better separation of concerns

## Documentation References

- [Voice AI Quickstart](https://docs.livekit.io/agents/quickstart/)
- [Tool Definition & Use](https://docs.livekit.io/agents/build/tools/)
- [Workflows](https://docs.livekit.io/agents/build/workflows/)
- [Metrics & Telemetry](https://docs.livekit.io/agents/build/metrics/)
- [SIP Transfers](https://docs.livekit.io/sip/transfer-cold/)

## Known Limitations

1. **Call Recording**: Temporarily disabled in new implementation - needs re-implementation
2. **Event Streaming**: Event handlers need to be updated for new session events
3. **Testing**: Comprehensive testing needed before production use

## Next Steps

1. Test the new implementation with sample calls
2. Re-enable call recording functionality
3. Add comprehensive event handlers for API streaming
4. Update frontend integration if needed
5. Performance testing and optimization

