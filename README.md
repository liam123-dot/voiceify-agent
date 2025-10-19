# Voiceify Voice Agent (Python)

A production-ready voice AI agent built with [LiveKit Agents Python SDK](https://docs.livekit.io/agents/) v1.2.0+. This agent handles inbound phone calls, processes voice interactions using configurable STT/LLM/TTS pipelines, and supports advanced features like agent transfers and call recording.

## Features

- **Configurable Voice Pipeline**: Support for multiple providers (OpenAI, Deepgram, ElevenLabs, etc.)
- **Agent Transfers**: Seamless handoffs between agents with preserved context
- **Phone Transfers**: Transfer calls to external numbers via SIP REFER
- **Call Recording**: Automatic recording with Supabase S3 storage
- **Real-time Events**: Stream conversation events to the API for monitoring
- **Function Tools**: Execute custom tools and API calls during conversations
- **Production Ready**: Docker containerization, comprehensive logging, error handling

## Architecture

```
voice-agent/
├── src/
│   ├── agent.py              # Main entry point & job handler
│   ├── agents/               # Agent management
│   │   ├── factory.py        # Create agents from config
│   │   └── registry.py       # Store agents for transfers
│   ├── tools/                # Tool execution & conversion
│   │   ├── converter.py      # Convert API tools to LiveKit format
│   │   └── transfer.py       # Transfer tool implementations
│   ├── session/              # Session lifecycle management
│   │   ├── events.py         # Event streaming to API
│   │   ├── recording.py      # Call recording with Egress
│   │   └── lifecycle.py      # Shutdown & first message handling
│   └── utils/                # Configuration & logging
│       ├── config.py         # Environment configuration
│       └── logging.py        # Structured logging
├── requirements.txt          # Python dependencies
├── Dockerfile               # Production container build
└── .env                     # Environment variables (create from .env.example)
```

## Setup

### Prerequisites

- Python 3.9 or higher
- LiveKit Cloud account or self-hosted LiveKit server
- API keys for AI providers (OpenAI, Deepgram, ElevenLabs)
- Voiceify API endpoint

### Installation

1. **Clone the repository and navigate to the agent directory:**

```bash
cd voice-agent
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Configure environment variables:**

```bash
cp .env.example .env
# Edit .env with your configuration
```

Required environment variables:
- `LIVEKIT_URL`: Your LiveKit server WebSocket URL
- `LIVEKIT_API_KEY`: LiveKit API key
- `LIVEKIT_API_SECRET`: LiveKit API secret
- `API_URL`: Voiceify API base URL
- `OPENAI_API_KEY`: OpenAI API key (for LLM)
- `DEEPGRAM_API_KEY`: Deepgram API key (for STT)
- `ELEVENLABS_API_KEY`: ElevenLabs API key (for TTS)

Optional (for call recording):
- `SUPABASE_PROJECT_REF`: Supabase project reference
- `SUPABASE_S3_ACCESS_KEY`: Supabase S3 access key
- `SUPABASE_S3_SECRET_KEY`: Supabase S3 secret key
- `SUPABASE_S3_REGION`: Supabase S3 region

### Running the Agent

**Development mode:**

```bash
python -m src.agent dev
```

**Production mode:**

```bash
python -m src.agent start
```

The agent will connect to LiveKit and wait for incoming jobs (calls).

## Docker Deployment

### Building the Image

```bash
docker build -t voiceify-agent:latest .
```

### Running with Docker

```bash
docker run --env-file .env voiceify-agent:latest
```

### Docker Compose (Example)

```yaml
version: '3.8'
services:
  voice-agent:
    build: .
    env_file: .env
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Configuration

### Agent Configuration

Agent configuration is fetched from the API based on the called phone number. The configuration includes:

- **Instructions**: System prompt defining agent behavior
- **Pipeline**: STT/LLM/TTS model configuration
- **Tools**: Available functions for the agent
- **Settings**: First message, noise cancellation, etc.

### Pipeline Configuration

The agent supports configurable pipelines with multiple providers:

**STT (Speech-to-Text):**
- Deepgram Nova 2
- Other Deepgram models

**LLM (Language Model):**
- OpenAI GPT-4, GPT-4 Turbo
- OpenAI GPT-3.5
- Other OpenAI models

**TTS (Text-to-Speech):**
- ElevenLabs (various models and voices)

### Tools

The agent supports custom tools defined in the API:

1. **API Tools**: Execute HTTP requests to external endpoints
2. **Transfer Tools**: 
   - Agent-to-agent transfers with context preservation
   - Phone number transfers via SIP REFER

## Event Streaming

The agent streams real-time events to the API:

- `room_connected`: Participant joined the room
- `user_input_transcribed`: User speech transcribed
- `speech_created`: Agent speech generated
- `function_tools_executed`: Tool execution completed
- `metrics_collected`: Performance metrics
- `transcript`: Full conversation history (on shutdown)
- `session_complete`: Session ended with recording URL

## Logging

Structured logging is configured with appropriate levels:

```python
[2024-01-15 10:30:00] [INFO] voiceify-agent: Connecting to room...
[2024-01-15 10:30:01] [INFO] voiceify-agent: Participant connected: sip_participant_123
[2024-01-15 10:30:02] [INFO] voiceify-agent: Agent loaded: Customer Support (agent-456)
```

## Troubleshooting

### Agent not receiving calls

1. Check LiveKit configuration (URL, API key, secret)
2. Verify Twilio SIP trunk configuration
3. Check agent worker logs for connection errors

### Tools not executing

1. Verify API_URL is set correctly
2. Check tool configuration in the database
3. Review tool execution logs

### Recording not working

1. Ensure all Supabase environment variables are set
2. Check S3 bucket permissions
3. Verify Egress is enabled in LiveKit

## Development

### Adding New Tools

Tools are automatically converted from the API format. To add custom tool types:

1. Add handler in `src/tools/converter.py`
2. Implement execution logic
3. Register with FunctionContext

### Extending Event Listeners

Add new event handlers in `src/session/events.py`:

```python
@assistant.on("custom_event")
def on_custom_event(event):
    # Handle event
    pass
```

## Resources

- [LiveKit Agents Documentation](https://docs.livekit.io/agents/)
- [LiveKit Python SDK](https://github.com/livekit/python-sdks)
- [Voiceify Documentation](https://docs.voiceify.com)

## License

MIT License - see LICENSE file for details

# voiceify-agent
