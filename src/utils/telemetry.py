"""
Telemetry setup for LiveKit agents.

Configures OpenTelemetry with Langfuse for tracing and monitoring agent sessions.
Based on: https://docs.livekit.io/agents/build/metrics/
"""

import base64
import os
import logging
from typing import Optional
from opentelemetry.util.types import AttributeValue

logger = logging.getLogger(__name__)


def setup_langfuse(
    metadata: Optional[dict[str, AttributeValue]] = None,
    *,
    host: Optional[str] = None,
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
):
    """
    Configure Langfuse tracer provider for OpenTelemetry.
    
    This enables telemetry data collection from LiveKit agents, including:
    - Session start traces
    - Agent turn traces
    - LLM node traces
    - Function tool traces
    - TTS node traces
    - End-of-turn detection traces
    - LLM and TTS metrics
    
    Args:
        metadata: Optional metadata to set as attributes on all spans
        host: Langfuse host URL (default: from LANGFUSE_HOST env var)
        public_key: Langfuse public key (default: from LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (default: from LANGFUSE_SECRET_KEY env var)
        
    Returns:
        TracerProvider instance or None if setup failed
        
    Example:
        >>> trace_provider = setup_langfuse(
        ...     metadata={"session.id": "my-session"},
        ...     host="https://cloud.langfuse.com",
        ...     public_key="pk-lf-...",
        ...     secret_key="sk-lf-..."
        ... )
    """
    try:
        from livekit.agents.telemetry import set_tracer_provider
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as e:
        logger.error(
            f"Failed to import OpenTelemetry dependencies: {e}. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp-proto-http"
        )
        return None
    
    # Get credentials from parameters or environment
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_HOST")
    
    # Validate required credentials
    if not public_key or not secret_key or not host:
        logger.warning(
            "Langfuse telemetry disabled: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, "
            "and LANGFUSE_HOST must be set"
        )
        return None
    
    try:
        # Configure OpenTelemetry exporter for Langfuse
        langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
        
        # Create and configure tracer provider
        trace_provider = TracerProvider()
        trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        
        # Set the tracer provider for LiveKit agents with optional metadata
        set_tracer_provider(trace_provider, metadata=metadata)
        
        logger.info(f"Langfuse telemetry enabled: {host}")
        return trace_provider
        
    except Exception as e:
        logger.error(f"Failed to setup Langfuse telemetry: {e}")
        return None

