"""
Call recording functionality using LiveKit Egress.

Handles starting call recordings and uploading them to Supabase S3 storage.
Based on: https://docs.livekit.io/agents/ops/recording/
"""

import logging
from typing import Optional, Tuple
from datetime import datetime
from livekit import api


logger = logging.getLogger(__name__)


async def start_call_recording(
    livekit_url: str,
    livekit_api_key: str,
    livekit_api_secret: str,
    organization_id: str,
    session_id: str,
    room_name: str,
    supabase_project_ref: str,
    supabase_s3_access_key: str,
    supabase_s3_secret_key: str,
    supabase_s3_region: str,
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Start recording a call using LiveKit Egress.
    
    Records the entire room conversation (audio + video) as MP4 and uploads it to Supabase S3 storage
    for later retrieval and analysis.
    
    Args:
        livekit_url: LiveKit server URL
        livekit_api_key: LiveKit API key
        livekit_api_secret: LiveKit API secret
        organization_id: Organization identifier for file path
        session_id: Session identifier for file path
        room_name: LiveKit room name to record
        supabase_project_ref: Supabase project reference
        supabase_s3_access_key: S3 access key
        supabase_s3_secret_key: S3 secret key
        supabase_s3_region: S3 region
        
    Returns:
        Tuple of (egress_id, recording_filename, started_at_unix_ms) or (None, None, None) on failure
        
    Example:
        >>> egress_id, filename = await start_call_recording(
        ...     livekit_url, api_key, api_secret, "org-123", "session-456", "room-789", ...
        ... )
        >>> if egress_id:
        ...     print(f"Recording started: {filename}")
    """
    try:
        # Generate unique filename with timestamp
        recording_filename = f"{organization_id}/{room_name}.mp4"
        
        logger.info(f"🎬 Starting call recording for room: {room_name}")
        logger.debug(f"📁 Recording filename: {recording_filename}")
        
        # Configure S3 upload to Supabase
        s3_endpoint = f"https://{supabase_project_ref}.supabase.co/storage/v1/s3"
        logger.debug(f"☁️  S3 endpoint: {s3_endpoint}")
        logger.debug(f"🗂️  S3 bucket: call-recordings")
        logger.debug(f"🌍 S3 region: {supabase_s3_region}")
        
        # Create the egress request using the new API
        req = api.RoomCompositeEgressRequest(
            room_name=room_name,
            file_outputs=[
                api.EncodedFileOutput(
                    file_type=api.EncodedFileType.MP4,
                    filepath=recording_filename,
                    s3=api.S3Upload(
                        bucket="call-recordings",
                        region=supabase_s3_region,
                        access_key=supabase_s3_access_key,
                        secret=supabase_s3_secret_key,
                        endpoint=s3_endpoint,
                        force_path_style=True,
                    ),
                )
            ],
        )
        
        # Create LiveKit API client and start recording
        logger.debug(f"🔗 Creating LiveKit API client: {livekit_url}")
        lkapi = api.LiveKitAPI(
            url=livekit_url,
            api_key=livekit_api_key,
            api_secret=livekit_api_secret,
        )
        
        logger.debug("📡 Starting room composite egress...")
        res = await lkapi.egress.start_room_composite_egress(req)
        await lkapi.aclose()
        
        egress_id = res.egress_id
        started_at_unix_ms = res.started_at  # Unix timestamp in milliseconds
        logger.info(f"✅ Call recording started successfully!")
        logger.info(f"   Egress ID: {egress_id}")
        logger.info(f"   Started At: {started_at_unix_ms}")
        logger.info(f"   Filename: {recording_filename}")
        logger.info(f"   Room: {room_name}")
        
        return egress_id, recording_filename, started_at_unix_ms
        
    except Exception as e:
        logger.error(f"❌ Failed to start call recording: {e}")
        logger.exception("Full error details:")
        return None, None, None


def get_recording_url(
    recording_filename: str,
    supabase_project_ref: str,
) -> str:
    """
    Generate a public URL for a recording file.
    
    Args:
        recording_filename: Path to the recording file in S3
        supabase_project_ref: Supabase project reference
        
    Returns:
        Public URL to access the recording
    """
    return (
        f"https://{supabase_project_ref}.supabase.co/storage/v1/object/public/"
        f"call-recordings/{recording_filename}"
    )

