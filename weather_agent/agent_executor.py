# ruff: noqa: E501
# pylint: disable=logging-fstring-interpolation
import logging

from typing import Any, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from weather_agent.weather_agent import WeatherAgent


logger = logging.getLogger(__name__)


class WeatherAgentExecutor(AgentExecutor):
    """WeatherAgentExecutor that uses the WeatherAgent to provide forecasts."""

    def __init__(self):
        """Initializes the WeatherAgentExecutor."""
        super().__init__()
        logger.info('Initializing WeatherAgentExecutor')
        self.agent = WeatherAgent()

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = context.get_user_input()
        task = context.current_task

        if not context.message:
            raise Exception('No message provided')

        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
            
        # invoke the underlying agent, using streaming results
        async for event in self.agent.stream(query, task.context_id):
            if event['is_task_complete']:
                await event_queue.enqueue_event(
                    TaskArtifactUpdateEvent(
                        append=False,
                        context_id=task.context_id,
                        task_id=task.id,
                        last_chunk=True,
                        artifact=new_text_artifact(
                            name='current_result',
                            description='Result of request to weather agent.',
                            text=event['content'],
                        ),
                    )
                )
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(state=TaskState.completed),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
            elif event['require_user_input']:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.input_required,
                            message=new_agent_text_message(
                                event['content'],
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=True,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )
            else:
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        status=TaskStatus(
                            state=TaskState.working,
                            message=new_agent_text_message(
                                event['content'],
                                task.context_id,
                                task.id,
                            ),
                        ),
                        final=False,
                        context_id=task.context_id,
                        task_id=task.id,
                    )
                )

    @override
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported')
