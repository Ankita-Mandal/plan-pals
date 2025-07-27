# ruff: noqa: E501
# pylint: disable=logging-fstring-interpolation
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import click
import uvicorn
from a2a.client.mcp_client import MCPClient
from a2a.server.agent_card import AgentCard
from a2a.server.agent_server import AgentServer
from a2a.server.app_context import AppContext
from a2a.server.request_handler import RequestHandler
from a2a.types import AgentCardData, AgentDescription
from dotenv import load_dotenv
from fastapi import FastAPI
from host_agent.agent_executor import HostAgentExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Default values
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 10000
DEFAULT_LOG_LEVEL = 'info'

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
required_env_vars = [
    'GOOGLE_GENAI_MODEL',
    'WEATHER_AGENT_URL',
    'EVENT_FINDER_AGENT_URL',
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]

if os.getenv('GOOGLE_GENAI_USE_VERTEXAI') == 'TRUE':
    vertex_vars = ['GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION']
    missing_vars.extend(
        [var for var in vertex_vars if not os.getenv(var)]
    )
else:
    if not os.getenv('GOOGLE_API_KEY'):
        missing_vars.append('GOOGLE_API_KEY')

if missing_vars:
    logger.error(
        f'Missing required environment variables: {", ".join(missing_vars)}'
    )
    logger.error(
        'Please set these variables in your .env file or environment.'
    )
    sys.exit(1)

# Create the agent executor
executor = HostAgentExecutor()

# Create the agent card
agent_card = AgentCard(
    data=AgentCardData(
        name_for_human='Weekend Wizard Host',
        name_for_model='WeekendWizardHost',
        description_for_human='A host agent that coordinates with weather and event finder agents to suggest weekend plans based on weather and user interests.',
        description_for_model=AgentDescription(
            description='I am the Weekend Wizard Host agent. I coordinate with the Weather Agent and Event Finder Agent to suggest weekend plans based on weather forecasts and available events. I can help users plan their weekend activities by considering weather conditions and their personal interests.',
            capabilities=[
                'Get weekend weather forecasts from the Weather Agent',
                'Find events and activities from the Event Finder Agent',
                'Suggest weekend plans based on weather and user interests',
                'Coordinate between multiple agents to provide comprehensive recommendations',
            ],
            limitations=[
                'I rely on other agents for weather and event information',
                'I cannot book tickets or make reservations',
                'My suggestions are based on available data and may not cover all possible activities',
            ],
        ),
        supported_content_types=['text', 'text/plain'],
    )
)


@asynccontextmanager
async def app_lifespan(app_context: AppContext):
    """Lifecycle management for the FastAPI application."""
    # Startup: initialize MCP client and register tools
    logger.info('Initializing MCP client...')
    mcp_client = MCPClient()
    
    # Register remote agents as tools
    weather_agent_url = os.getenv('WEATHER_AGENT_URL')
    event_finder_agent_url = os.getenv('EVENT_FINDER_AGENT_URL')
    
    logger.info(f'Registering Weather Agent at {weather_agent_url}')
    weather_agent = await mcp_client.register_agent(weather_agent_url)
    
    logger.info(f'Registering Event Finder Agent at {event_finder_agent_url}')
    event_finder_agent = await mcp_client.register_agent(event_finder_agent_url)
    
    # Store agents in app context for use by the executor
    app_context.set('weather_agent', weather_agent)
    app_context.set('event_finder_agent', event_finder_agent)
    
    logger.info('Host Agent startup complete')
    yield
    
    # Shutdown
    logger.info('Shutting down Host Agent...')
    await mcp_client.close()
    logger.info('Host Agent shutdown complete')


@click.command()
@click.option('--host', default=DEFAULT_HOST, help='Host to bind the server to')
@click.option(
    '--port', default=DEFAULT_PORT, type=int, help='Port to bind the server to'
)
@click.option(
    '--log-level',
    default=DEFAULT_LOG_LEVEL,
    help='Log level (debug, info, warning, error, critical)',
)
def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    log_level: str = DEFAULT_LOG_LEVEL,
):
    """Main entry point for the Host Agent server."""
    # Set log level
    logging.getLogger().setLevel(log_level.upper())
    
    # Create app context
    app_context = AppContext()
    
    # Create request handler
    request_handler = RequestHandler(executor=executor, app_context=app_context)
    
    # Create agent server
    agent_server = AgentServer(
        request_handler=request_handler, agent_card=agent_card
    )
    
    # Create FastAPI app
    app = FastAPI(lifespan=lambda app: app_lifespan(app_context))
    
    # Include agent server routes
    app.include_router(agent_server.router)
    
    # Run the server
    async def run_server_async():
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            log_level=log_level.lower(),
        )
        
        async with app_lifespan(app_context):
            logger.info(f'Starting Host Agent server on {host}:{port}')
            server = uvicorn.Server(config)
            await server.serve()
    
    # Run the server
    asyncio.run(run_server_async())


if __name__ == '__main__':
    main()
