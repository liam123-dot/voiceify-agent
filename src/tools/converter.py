"""
Tool converter for the new LiveKit Agents API.

Converts API tools into function_tool decorated methods that can be added to Agent classes.
Based on: https://docs.livekit.io/agents/build/tools/
"""

import logging
import json
from typing import Any, Dict, List, Callable
import aiohttp

from livekit.agents import function_tool, RunContext, llm
from livekit.protocol.sip import TransferSIPParticipantRequest
from livekit import api

from ..agents.registry import AgentRegistry


logger = logging.getLogger(__name__)


def create_tool_functions(
    api_tools: List[Dict[str, Any]],
    agent_registry: AgentRegistry,
    caller_phone_number: str,
    called_phone_number: str,
    api_base_url: str,
    livekit_url: str,
    livekit_api_key: str,
    livekit_api_secret: str,
) -> List[Callable]:
    """
    Convert API tools to function_tool decorated functions.
    
    Creates functions that can be added as methods to Agent classes.
    
    Args:
        api_tools: List of tool definitions from the API
        agent_registry: Registry containing pre-instantiated agents for transfers
        caller_phone_number: Phone number of the caller
        called_phone_number: Phone number that was called
        api_base_url: Base URL for API tool execution endpoint
        livekit_url: LiveKit server URL for SIP operations
        livekit_api_key: LiveKit API key
        livekit_api_secret: LiveKit API secret
        
    Returns:
        List of decorated tool functions
    """
    tool_functions = []
    
    for tool in api_tools:
        tool_id = tool.get("id")
        tool_name = tool.get("name")
        tool_type = tool.get("type")
        tool_description = tool.get("description") or tool.get("label") or f"Tool: {tool_name}"
        tool_parameters = tool.get("parameters", {"type": "object", "properties": {}})
        
        logger.debug(f"Converting tool: {tool_name} (type: {tool_type})")
        
        # Handle transfer_call tools
        if tool_type == "transfer_call":
            static_config = tool.get("staticConfig", {})
            target = static_config.get("target", {})
            target_type = target.get("type")
            
            if target_type == "agent":
                # Agent-to-agent transfer
                target_agent_id = target.get("agentId")
                
                @function_tool(name=tool_name, description=tool_description)
                async def agent_transfer_tool(context: RunContext):
                    """Transfer to another agent."""
                    target_agent = agent_registry.get(target_agent_id)
                    if not target_agent:
                        raise llm.ToolError(f"Target agent {target_agent_id} not found")
                    
                    logger.info(f"Transferring to agent: {target_agent_id}")
                    # Return the agent instance to trigger handoff
                    return target_agent, f"Transferring to {target_agent._voiceify_agent_name}"
                
                tool_functions.append(agent_transfer_tool)
                
            elif target_type == "number":
                # Phone number transfer using SIP REFER
                target_number = target.get("phoneNumber")
                
                @function_tool(name=tool_name, description=tool_description)
                async def phone_transfer_tool(context: RunContext):
                    """Transfer call to a phone number."""
                    from livekit.agents import get_job_context
                    
                    try:
                        job_ctx = get_job_context()
                        room = job_ctx.room
                        
                        # Find the SIP participant
                        # SIP participants have identity starting with "sip_" or have "sip." in attributes
                        sip_participant = None
                        
                        # Check remote participants
                        for participant in room.remote_participants.values():
                            # Check if participant is SIP type by identity or attributes
                            if participant.identity.startswith("sip_") or \
                               "sip.phoneNumber" in participant.attributes or \
                               "sip.trunkPhoneNumber" in participant.attributes:
                                sip_participant = participant
                                logger.info(f"Found SIP participant: {participant.identity}")
                                break
                        
                        if not sip_participant:
                            # Log available participants for debugging
                            participant_info = [
                                f"{p.identity} (attrs: {list(p.attributes.keys())})" 
                                for p in room.remote_participants.values()
                            ]
                            logger.error(f"No SIP participant found. Available participants: {participant_info}")
                            raise llm.ToolError("No SIP participant found to transfer")
                        
                        logger.info(f"Transferring call to: {target_number}")
                        
                        # Create LiveKit API client
                        async with api.LiveKitAPI(
                            url=livekit_url,
                            api_key=livekit_api_key,
                            api_secret=livekit_api_secret,
                        ) as lk_api:
                            # Transfer the call using SIP REFER
                            transfer_request = TransferSIPParticipantRequest(
                                participant_identity=sip_participant.identity,
                                room_name=room.name,
                                transfer_to=f"tel:{target_number}",
                                play_dialtone=False,
                            )
                            
                            await lk_api.sip.transfer_sip_participant(transfer_request)
                            logger.info(f"Successfully initiated transfer to {target_number}")
                            
                            return f"Transferring your call to {target_number}"
                            
                    except Exception as e:
                        logger.error(f"Error transferring call: {e}")
                        raise llm.ToolError(f"Failed to transfer call: {str(e)}")
                
                tool_functions.append(phone_transfer_tool)
            else:
                logger.warning(f"Unknown transfer target type: {target_type}")
        
        else:
            # Standard API tool execution
            # Create function using factory pattern like the LiveKit examples
            def make_api_tool(tool_config):
                tool_id = tool_config.get("id")
                tool_name = tool_config.get("name")
                tool_description_inner = tool_config.get("description") or tool_config.get("label") or f"Tool: {tool_name}"
                static_config = tool_config.get("staticConfig", {})
                tool_params = tool_config.get("parameters", {"type": "object", "properties": {}})
                
                # Build raw schema matching LiveKit's RawFunctionDescription format
                # Note: raw_schema should NOT include "type" field
                raw_schema = {
                    "name": tool_name,
                    "description": tool_description_inner,
                    "parameters": tool_params,
                }
                print(f"Raw schema: {raw_schema}")
                
                # Create handler function with tool-specific params captured in closure
                # For raw_schema tools, signature: (raw_arguments, context)
                def make_handler(tid, tname, sconfig):
                    async def handler(raw_arguments: dict[str, object], context: RunContext):
                        """Execute API tool with dynamic parameters."""
                        return await _execute_api_tool(
                            tool_id=tid,
                            tool_name=tname,
                            params=raw_arguments,  # raw_arguments contains the actual parameter values
                            static_config=sconfig,
                            api_base_url=api_base_url,
                            caller_phone_number=caller_phone_number,
                            called_phone_number=called_phone_number,
                        )
                    return handler
                
                handler = make_handler(tool_id, tool_name, static_config)
                
                # Create tool using function_tool with raw_schema
                # This matches the pattern from LiveKit's dynamic tool creation example
                return function_tool(handler, raw_schema=raw_schema)
            
            tool_functions.append(make_api_tool(tool))
    
    logger.info(f"Created {len(tool_functions)} tool functions")
    return tool_functions


async def _execute_api_tool(
    tool_id: str,
    tool_name: str,
    params: Dict[str, Any],
    static_config: Dict[str, Any],
    api_base_url: str,
    caller_phone_number: str,
    called_phone_number: str,
) -> str:
    """
    Execute an API tool by making an HTTP request.
    
    This helper function is used by dynamically created tools.
    """
    try:
        # Build request payload
        payload = {
            "toolId": tool_id,
            "parameters": params,
            "staticConfig": static_config,
            "metadata": {
                "callerPhoneNumber": caller_phone_number,
                "calledPhoneNumber": called_phone_number,
            }
        }
        
        logger.info(f"Executing API tool: {tool_name}")
        logger.debug(f"Tool payload: {payload}")
        
        # Execute tool via API
        endpoint = f"{api_base_url}/api/tools/{tool_id}/execute"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if not response.ok:
                    error_text = await response.text()
                    logger.error(f"Tool execution failed: {response.status} {error_text}")
                    raise llm.ToolError(f"Tool execution failed: {error_text}")
                
                result = await response.json()
                logger.info(f"Tool executed successfully: {tool_name}")
                
                # Return the result
                if isinstance(result, dict):
                    return json.dumps(result)
                return str(result)
                
    except aiohttp.ClientError as e:
        logger.error(f"Network error executing tool {tool_name}: {e}")
        raise llm.ToolError(f"Network error: {str(e)}")
    except Exception as e:
        logger.error(f"Error executing tool {tool_name}: {e}")
        raise llm.ToolError(f"Unexpected error: {str(e)}")
