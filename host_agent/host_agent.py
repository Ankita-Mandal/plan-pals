# ruff: noqa: E501
# pylint: disable=logging-fstring-interpolation
import logging
import os
import json
from collections.abc import AsyncIterable
from typing import Any, Dict, List, Literal, Optional

from a2a.client.agent import Agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables.config import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO)

memory = MemorySaver()


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class HostAgent:
    """Host Agent that coordinates with Weather and Event Finder agents to suggest weekend plans."""

    SYSTEM_INSTRUCTION = """You are the PlanPals Host, an assistant that helps users plan their weekend activities. 
    
    Your primary functions are:
    1. Coordinate with the Weather Agent to get weekend weather forecasts
    2. Coordinate with the Event Finder Agent to discover events and activities
    3. Suggest personalized weekend plans based on weather conditions and user interests
    
    When a user asks for weekend plans:
    - First, get the weather forecast for the upcoming weekend using the get_weather_forecast tool
    - Then, find relevant events using the search_events tool
    - Finally, provide personalized recommendations that consider both weather and available events
    
    Always format your responses in Markdown, with clear sections for Weather, Events, and Recommendations.
    Be friendly, helpful, and enthusiastic about helping users plan a great weekend!
    """

    RESPONSE_FORMAT_INSTRUCTION: str = (
        'Select status as "completed" if the request is fully addressed and no further input is needed. '
        'Select status as "input_required" if you need more information from the user or are asking a clarifying question. '
        'Select status as "error" if an error occurred or the request cannot be fulfilled.'
    )

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self, weather_agent: Agent, event_finder_agent: Agent):
        """Initializes the Host Agent with connections to other agents."""
        logger.info('Initializing HostAgent...')
        self.weather_agent = weather_agent
        self.event_finder_agent = event_finder_agent
        
        try:
            model = os.getenv('GOOGLE_GENAI_MODEL')
            if not model:
                raise ValueError(
                    'GOOGLE_GENAI_MODEL environment variable is not set'
                )

            if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') == 'TRUE':
                # If using Vertex AI, initialize with Vertex AI
                logger.info('ChatVertexAI model initialized successfully.')
                self.model = ChatVertexAI(model=model)
            else:
                # Using Google Generative AI
                self.model = ChatGoogleGenerativeAI(model=model)
                logger.info(
                    'ChatGoogleGenerativeAI model initialized successfully.'
                )

        except Exception as e:
            logger.error(
                f'Failed to initialize ChatGoogleGenerativeAI model: {e}',
                exc_info=True,
            )
            raise

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        logger.info(
            f"HostAgent.ainvoke called with query: '{query}', session_id: '{session_id}'"
        )
        try:
            host_agent_runnable = create_react_agent(
                self.model,
                tools=self._get_tools(),
                checkpointer=memory,
                prompt=self.SYSTEM_INSTRUCTION,
                response_format=(
                    self.RESPONSE_FORMAT_INSTRUCTION,
                    ResponseFormat,
                ),
            )
            logger.debug(
                'LangGraph React agent for Host task created/configured.'
            )

            config: RunnableConfig = {'configurable': {'thread_id': session_id}}
            langgraph_input = {'messages': [('user', query)]}

            logger.debug(
                f'Invoking Host Agent with input: {langgraph_input} and config: {config}'
            )
            result = await host_agent_runnable.ainvoke(langgraph_input, config)
            logger.debug(f'Raw result from Host Agent: {result}')

            # Extract the response from the state
            response = self._get_agent_response_from_state(config, host_agent_runnable)
            logger.info(
                f'Final response from Host Agent for session {session_id}: {response}'
            )
            return response

        except Exception as e:
            logger.error(
                f'Error during HostAgent.ainvoke for session {session_id}: {e}',
                exc_info=True,
            )
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'An error occurred: {str(e)}',
            }

    def _get_tools(self):
        """Returns the tools for the Host Agent, including access to other agents."""
        
        async def get_weather_forecast(location: str) -> Dict[str, Any]:
            """Get the weather forecast for the upcoming weekend at the specified location."""
            try:
                logger.info(f"Requesting weather forecast for {location}")
                response = await self.weather_agent.ainvoke(
                    f"What's the weather forecast for this weekend in {location}?"
                )
                
                # Extract content from the response
                if isinstance(response, dict) and 'content' in response:
                    content = response['content']
                else:
                    content = str(response)
                    
                logger.info(f"Weather forecast response received: {content[:100]}...")
                return {
                    "location": location,
                    "forecast": content
                }
            except Exception as e:
                logger.error(f"Error getting weather forecast: {e}", exc_info=True)
                return {"error": f"Failed to get weather forecast: {str(e)}"}
        
        async def search_events(location: str, category: str = "all") -> Dict[str, Any]:
            """Search for events in a specific location for the upcoming weekend."""
            try:
                logger.info(f"Searching for {category} events in {location}")
                query = f"Find {category} events in {location} for this weekend"
                if category == "all":
                    query = f"Find events in {location} for this weekend"
                    
                response = await self.event_finder_agent.ainvoke(query)
                
                # Extract content from the response
                if isinstance(response, dict) and 'content' in response:
                    content = response['content']
                else:
                    content = str(response)
                    
                logger.info(f"Event search response received: {content[:100]}...")
                return {
                    "location": location,
                    "category": category,
                    "events": content
                }
            except Exception as e:
                logger.error(f"Error searching for events: {e}", exc_info=True)
                return {"error": f"Failed to search for events: {str(e)}"}
        
        async def suggest_weekend_plan(location: str, interests: List[str] = None, indoor_only: bool = False) -> Dict[str, Any]:
            """Suggest a weekend plan based on weather, available events, and user interests."""
            try:
                logger.info(f"Generating weekend plan for {location} with interests: {interests}")
                
                # Get weather forecast
                weather_response = await get_weather_forecast(location)
                
                # Determine categories based on interests
                category = "all"
                if interests and len(interests) > 0:
                    # Map interests to categories (simplified mapping)
                    category_mapping = {
                        "music": ["music", "concert", "festival", "band", "jazz", "rock"],
                        "food": ["food", "dining", "restaurant", "culinary", "cooking", "eat"],
                        "sports": ["sports", "athletic", "game", "match", "run", "marathon"],
                        "arts": ["art", "museum", "gallery", "exhibition", "theater", "culture"],
                        "family": ["family", "kids", "children", "parents", "educational"]
                    }
                    
                    # Find matching categories
                    matching_categories = []
                    for interest in interests:
                        interest_lower = interest.lower()
                        for cat, keywords in category_mapping.items():
                            if interest_lower in keywords or any(keyword in interest_lower for keyword in keywords):
                                matching_categories.append(cat)
                    
                    # If we found matching categories, use the first one
                    if matching_categories:
                        category = matching_categories[0]
                
                # Get events
                events_response = await search_events(location, category)
                
                # Combine results
                return {
                    "location": location,
                    "weather": weather_response.get("forecast", "Weather information unavailable"),
                    "events": events_response.get("events", "Event information unavailable"),
                    "interests": interests or ["general"],
                    "indoor_only": indoor_only
                }
            except Exception as e:
                logger.error(f"Error suggesting weekend plan: {e}", exc_info=True)
                return {"error": f"Failed to suggest weekend plan: {str(e)}"}
        
        return [get_weather_forecast, search_events, suggest_weekend_plan]

    def _get_agent_response_from_state(
        self, config: RunnableConfig, agent_runnable
    ) -> dict[str, Any]:
        """Retrieves and formats the agent's response from the state of the given agent_runnable."""
        try:
            state = agent_runnable.get_state(config)
            logger.debug(f'Agent state for config {config}: {state}')

            # Extract values from state
            state_values = state.values
            if not state_values:
                logger.warning(f'Empty state values for config {config}')
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': 'I couldn\'t process your request. Could you please provide more details?',
                }

            # Try to find the structured response in the state
            messages = state_values.get('messages', [])
            if not messages:
                logger.warning(f'No messages found in state for config {config}')
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': 'I couldn\'t process your request. Could you please try again?',
                }

            # Look for the most recent AI message with a structured response
            for message in reversed(messages):
                if message[0] == 'ai' and isinstance(message[1], dict):
                    response_format = message[1].get('response_format')
                    if response_format:
                        status = response_format.get('status')
                        message_content = response_format.get('message', '')

                        if status == 'completed':
                            return {
                                'is_task_complete': True,
                                'require_user_input': False,
                                'content': message_content,
                            }
                        elif status == 'input_required':
                            return {
                                'is_task_complete': False,
                                'require_user_input': True,
                                'content': message_content,
                            }
                        elif status == 'error':
                            return {
                                'is_task_complete': True,
                                'require_user_input': False,
                                'content': f'Error: {message_content}',
                            }

                # Fallback to looking for regular AI messages
                elif message[0] == 'ai' and isinstance(message[1], AIMessage):
                    ai_message = message[1]
                    return {
                        'is_task_complete': True,
                        'require_user_input': False,
                        'content': ai_message.content,
                    }

            # If we reach here, we couldn't find a suitable response
            logger.warning(
                f'No suitable AI message found in state for config {config}'
            )
            
            # Last resort: try to extract content from the last message
            last_message = messages[-1] if messages else None
            if last_message and last_message[0] == 'ai':
                ai_content = last_message[1]
                if isinstance(ai_content, dict) and 'content' in ai_content:
                    return {
                        'is_task_complete': True,
                        'require_user_input': False,
                        'content': ai_content['content'],
                    }
                elif isinstance(ai_content, list):
                    text_parts = [
                        part['text']
                        for part in ai_content
                        if isinstance(part, dict) and part.get('type') == 'text'
                    ]
                    if text_parts:
                        logger.warning(
                            f'Structured response not found. Falling back to concatenated text from last AI message parts for config {config}.'
                        )
                        return {
                            'is_task_complete': True,
                            'require_user_input': False,
                            'content': '\n'.join(text_parts),
                        }

            logger.warning(
                f'Structured response not found or not in expected format, and no suitable fallback AI message. State for config {config}: {state_values}'
            )
            return {
                'is_task_complete': False,
                'require_user_input': True,
                'content': 'We are unable to process your request at the moment due to an unexpected response format. Please try again.',
            }
        except Exception as e:
            logger.error(f'Error extracting response from state: {e}', exc_info=True)
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'An error occurred while processing your request: {str(e)}',
            }

    async def stream(self, query: str, session_id: str) -> AsyncIterable[Any]:
        logger.info(
            f"HostAgent.stream called with query: '{query}', sessionId: '{session_id}'"
        )
        agent_runnable = create_react_agent(
            self.model,
            tools=self._get_tools(),
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(
                self.RESPONSE_FORMAT_INSTRUCTION,
                ResponseFormat,
            ),
        )
        config: RunnableConfig = {'configurable': {'thread_id': session_id}}
        langgraph_input = {'messages': [('user', query)]}

        logger.debug(
            f'Streaming from Host Agent with input: {langgraph_input} and config: {config}'
        )
        try:
            async for chunk in agent_runnable.astream_events(
                langgraph_input, config, version='v1'
            ):
                logger.debug(f'Stream chunk for {session_id}: {chunk}')
                event_name = chunk.get('event')
                data = chunk.get('data', {})
                content_to_yield = None

                if event_name == 'on_tool_start':
                    tool_name = data.get('name', 'a tool')
                    content_to_yield = f'Using tool: {tool_name}...'
                elif event_name == 'on_chat_model_stream':
                    message_chunk = data.get('chunk')
                    if (
                        isinstance(message_chunk, AIMessageChunk)
                        and message_chunk.content
                    ):
                        content_to_yield = message_chunk.content

                if content_to_yield:
                    yield {
                        'is_task_complete': False,
                        'require_user_input': False,
                        'content': content_to_yield,
                    }

            # After all events, get the final structured response from the agent's state
            final_response = self._get_agent_response_from_state(
                config, agent_runnable
            )
            logger.info(
                f'Final response from state after stream for session {session_id}: {final_response}'
            )
            yield final_response

        except Exception as e:
            logger.error(
                f'Error during HostAgent.stream for session {session_id}: {e}',
                exc_info=True,
            )
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'An error occurred during streaming: {getattr(e, "message", str(e))}',
            }
