"""
Event streaming to the Voiceify API.

Handles sending real-time session events to the API for monitoring,
analytics, and transcript storage.
"""

import logging
from typing import Any, Dict, Optional
from datetime import datetime
import aiohttp


logger = logging.getLogger(__name__)


async def send_event_to_api(
    agent_id: str,
    twilio_call_sid: str,
    caller_phone_number: str,
    room_name: str,
    session_id: str,
    event_type: str,
    data: Any,
    api_url: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Send an event to the Voiceify API.
    
    Events are sent asynchronously and failures are logged but don't interrupt
    the agent operation.
    
    Args:
        agent_id: Unique identifier for the agent
        twilio_call_sid: Twilio call SID
        caller_phone_number: Phone number of the caller
        room_name: LiveKit room name
        session_id: Unique session identifier
        event_type: Type of event being sent
        data: Event payload data
        api_url: Base URL for the API
        metadata: Optional additional metadata
    """
    try:
        payload = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data": data,
            "sessionId": session_id,
            "roomName": room_name,
            "twilioCallSid": twilio_call_sid,
            "callerPhoneNumber": caller_phone_number,
        }
        
        if metadata:
            payload["metadata"] = metadata
        
        endpoint = f"{api_url}/api/agents/{agent_id}/calls"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    logger.error(
                        f"Failed to send {event_type} event: "
                        f"{response.status} {error_text}"
                    )
                else:
                    logger.debug(f"Sent {event_type} event to API")
                    
    except Exception as e:
        logger.error(f"Error sending {event_type} event: {e}")

