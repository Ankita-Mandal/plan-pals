# ruff: noqa: E501
# pylint: disable=logging-fstring-interpolation
import logging
import os
import requests
from datetime import datetime, timedelta
from collections.abc import AsyncIterable
from typing import Any, Literal

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


class WeatherAgent:
    """Weather Agent for providing weekend weather forecasts."""

    SYSTEM_INSTRUCTION = """You are a specialized assistant for providing weather forecasts. Your primary function is to utilize the provided tools to check weather predictions for the upcoming weekend and answer related questions. You must rely exclusively on these tools for information; do not invent weather data. Ensure that your Markdown-formatted response includes all relevant weather information, focusing on weekend days (Saturday and Sunday)."""

    RESPONSE_FORMAT_INSTRUCTION: str = (
        'Select status as "completed" if the request is fully addressed and no further input is needed. '
        'Select status as "input_required" if you need more information from the user or are asking a clarifying question. '
        'Select status as "error" if an error occurred or the request cannot be fulfilled.'
    )

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']

    def __init__(self):
        """Initializes the Weather agent."""
        logger.info('Initializing WeatherAgent...')
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

        # Initialize tools for weather forecasting
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weekend_weather",
                    "description": "Get the weather forecast for the upcoming weekend for a specific location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state/country for the weather forecast"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]

    async def ainvoke(self, query: str, session_id: str) -> dict[str, Any]:
        logger.info(
            f"WeatherAgent.ainvoke called with query: '{query}', session_id: '{session_id}'"
        )
        try:
            weather_agent_runnable = create_react_agent(
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
                'LangGraph React agent for Weather task created/configured.'
            )

            config: RunnableConfig = {'configurable': {'thread_id': session_id}}
            langgraph_input = {'messages': [('user', query)]}

            logger.debug(
                f'Invoking Weather Agent with input: {langgraph_input} and config: {config}'
            )
            result = await weather_agent_runnable.ainvoke(langgraph_input, config)
            logger.debug(f'Raw result from Weather Agent: {result}')

            # Extract the response from the state
            response = self._get_agent_response_from_state(config, weather_agent_runnable)
            logger.info(
                f'Final response from Weather Agent for session {session_id}: {response}'
            )
            return response

        except Exception as e:
            logger.error(
                f'Error during WeatherAgent.ainvoke for session {session_id}: {e}',
                exc_info=True,
            )
            return {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'An error occurred: {str(e)}',
            }

    def _get_tools(self):
        """Returns the tools for the Weather Agent."""
        
        def get_weekend_weather(location: str):
            """Get the weather forecast for the upcoming weekend."""
            try:
                # Calculate the upcoming weekend dates
                today = datetime.now()
                days_until_saturday = (5 - today.weekday()) % 7
                if days_until_saturday == 0:
                    days_until_saturday = 7
                saturday = today + timedelta(days=days_until_saturday)
                sunday = saturday + timedelta(days=1)
                
                # In a real implementation, you would call a weather API here
                # For now, we'll return mock data
                weekend_forecast = {
                    "location": location,
                    "forecast": [
                        {
                            "date": saturday.strftime("%Y-%m-%d"),
                            "day": "Saturday",
                            "condition": "Partly Cloudy",
                            "temperature_high": 72,
                            "temperature_low": 58,
                            "precipitation_chance": 20,
                            "humidity": 65,
                            "wind_speed": 10
                        },
                        {
                            "date": sunday.strftime("%Y-%m-%d"),
                            "day": "Sunday",
                            "condition": "Sunny",
                            "temperature_high": 75,
                            "temperature_low": 60,
                            "precipitation_chance": 10,
                            "humidity": 60,
                            "wind_speed": 8
                        }
                    ]
                }
                
                return weekend_forecast
            except Exception as e:
                return {"error": f"Failed to get weather forecast: {str(e)}"}
        
        return [get_weekend_weather]

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
            f"WeatherAgent.stream called with query: '{query}', sessionId: '{session_id}'"
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
            f'Streaming from Weather Agent with input: {langgraph_input} and config: {config}'
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
                f'Error during WeatherAgent.stream for session {session_id}: {e}',
                exc_info=True,
            )
            yield {
                'is_task_complete': True,
                'require_user_input': False,
                'content': f'An error occurred during streaming: {getattr(e, "message", str(e))}',
            }
