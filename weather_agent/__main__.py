#!/usr/bin/env python3
# ruff: noqa: E501
# pylint: disable=logging-fstring-interpolation

import asyncio
import os
import sys

from contextlib import asynccontextmanager
from typing import Any

import click
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentCardData,
    AgentDescription,
    AgentSkill,
)
from weather_agent.agent_executor import WeatherAgentExecutor
from weather_agent.weather_agent import WeatherAgent
from dotenv import load_dotenv


load_dotenv(override=True)

app_context: dict[str, Any] = {}


DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 10001
DEFAULT_LOG_LEVEL = 'info'


@asynccontextmanager
async def app_lifespan(context: dict[str, Any]):
    """Manages the lifecycle of shared resources."""
    print('Lifespan: Initializing Weather Agent...')

    try:
        # Initialize any resources here
        yield  # Application runs here
    except Exception as e:
        print(f'Lifespan: Error during initialization: {e}', file=sys.stderr)
        raise
    finally:
        print('Lifespan: Shutting down Weather Agent...')
        # Clear the application context
        print('Lifespan: Clearing application context.')
        context.clear()


def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_level: str = DEFAULT_LOG_LEVEL,
):
    """Command Line Interface to start the Weather Agent server."""
    # Check for required environment variables
    if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') != 'TRUE' and not os.getenv(
        'GOOGLE_API_KEY'
    ):
        raise ValueError(
            'GOOGLE_API_KEY environment variable not set and '
            'GOOGLE_GENAI_USE_VERTEXAI is not TRUE.'
        )

    async def run_server_async():
        async with app_lifespan(app_context):
            # Initialize WeatherAgentExecutor
            weather_agent_executor = WeatherAgentExecutor()

            request_handler = DefaultRequestHandler(
                agent_executor=weather_agent_executor,
                task_store=InMemoryTaskStore(),
            )

            # Create the agent card
            agent_card = AgentCard(
                data=AgentCardData(
                    name_for_human='PlanPals Weather Agent',
                    name_for_model='PlanPalsWeatherAgent',
                    description_for_human='A weather agent that provides forecasts for the upcoming weekend.',
                    description_for_model=AgentDescription(
                        description='I am the PlanPals Weather Agent. I provide weather forecasts for the upcoming weekend to help users plan their activities.',
                        capabilities=[
                            'Provide weather forecasts for the upcoming weekend',
                            'Give temperature, precipitation, and general weather conditions',
                            'Offer weather-appropriate activity suggestions',
                        ],
                        limitations=[
                            'I use mocked weather data for demonstration purposes',
                            'I cannot provide real-time weather alerts',
                            'My forecasts are limited to the upcoming weekend',
                        ],
                    ),
                    supported_content_types=['text', 'text/plain'],
                )
            )

            # Create the A2AServer instance
            a2a_server = A2AStarletteApplication(
                agent_card=agent_card,
                http_handler=request_handler,
            )

            # Get the ASGI app from the A2AServer instance
            asgi_app = a2a_server.build()

            config = uvicorn.Config(
                app=asgi_app,
                host=host,
                port=port,
                log_level=log_level.lower(),
                lifespan='auto',
            )

            uvicorn_server = uvicorn.Server(config)

            print(
                f'Starting Weather Agent server at http://{host}:{port} with log-level {log_level}...'
            )
            try:
                await uvicorn_server.serve()
            except KeyboardInterrupt:
                print('Server shutdown requested (KeyboardInterrupt).')
            finally:
                print('Weather Agent server has stopped.')

    try:
        asyncio.run(run_server_async())
    except RuntimeError as e:
        if 'cannot be called from a running event loop' in str(e):
            print(
                'Critical Error: Attempted to nest asyncio.run(). This should have been prevented.',
                file=sys.stderr,
            )
        else:
            print(f'RuntimeError in main: {e}', file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f'An unexpected error occurred in main: {e}', file=sys.stderr)
        sys.exit(1)


@click.command()
@click.option(
    '--host',
    'host',
    default=DEFAULT_HOST,
    help='Hostname to bind the server to.',
)
@click.option(
    '--port',
    'port',
    default=DEFAULT_PORT,
    type=int,
    help='Port to bind the server to.',
)
@click.option(
    '--log-level',
    'log_level',
    default=DEFAULT_LOG_LEVEL,
    help='Uvicorn log level.',
)
def cli(host: str, port: int, log_level: str):
    main(host, port, log_level)


if __name__ == '__main__':
    main()
