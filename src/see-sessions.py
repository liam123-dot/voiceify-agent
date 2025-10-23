#!/usr/bin/env python3
"""
LiveKit Session Manager - List and kill active sessions.
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from livekit import api


async def get_client(env_file=".env.staging"):
    """Create LiveKit API client."""
    load_dotenv(env_file)
    
    url = os.getenv("LIVEKIT_URL")
    key = os.getenv("LIVEKIT_API_KEY")
    secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not all([url, key, secret]):
        print("Error: Missing LiveKit configuration in environment")
        return None
    
    return api.LiveKitAPI(url, key, secret)


async def list_sessions(env_file=".env.staging"):
    """List all active sessions."""
    client = await get_client(env_file)
    if not client:
        return []
    
    try:
        response = await client.room.list_rooms(api.ListRoomsRequest())
        rooms = list(response.rooms)
        await client.aclose()
        return rooms
    except Exception as e:
        print(f"Error listing sessions: {e}")
        await client.aclose()
        return []


async def kill_session(room_name: str, env_file=".env.staging"):
    """Kill a session by room name."""
    client = await get_client(env_file)
    if not client:
        return False
    
    try:
        await client.room.delete_room(api.DeleteRoomRequest(room=room_name))
        await client.aclose()
        return True
    except Exception as e:
        print(f"Error killing session: {e}")
        await client.aclose()
        return False


def format_duration(seconds: float) -> str:
    """Format duration."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{secs}s"


async def main_async():
    """Main async function."""
    env_file = ".env.local"
    args = sys.argv[1:]
    
    # Check for --env flag
    if "--env" in args:
        idx = args.index("--env")
        if idx + 1 < len(args):
            env_file = args[idx + 1]
            args.pop(idx)
            args.pop(idx)
    
    # If room name provided, kill it
    if args:
        room_name = args[0]
        print(f"Killing session: {room_name}")
        success = await kill_session(room_name, env_file)
        if success:
            print("✓ Session killed")
        else:
            print("✗ Failed to kill session")
        return
    
    # Otherwise, list sessions
    rooms = await list_sessions(env_file)
    
    if not rooms:
        print("No active sessions")
        return
    
    print(f"\nActive Sessions ({len(rooms)}):\n")
    for idx, room in enumerate(rooms, 1):
        duration = datetime.now().timestamp() - room.creation_time
        print(f"  [{idx}] {room.name}")
        print(f"      Duration: {format_duration(duration)}, Participants: {room.num_participants}")
    
    print(f"\nTo kill a session: python src/see-sessions.py <room_name>")


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
