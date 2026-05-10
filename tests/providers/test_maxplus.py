from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from providers.base import ProviderConfig
from providers.maxplus import MAXPLUS_MODEL_IDS, MaxPlusProvider


class MockMessage:
    def __init__(self, role, content):
        self.role = role
        self.content = content


class MockRequest:
    def __init__(self, model="gpt-5.4-mini"):
        self.model = model
        self.messages = [MockMessage("user", "say pong")]
        self.max_tokens = 32
        self.temperature = None
        self.top_p = None
        self.system = None
        self.stop_sequences = None
        self.tools = []
        self.tool_choice = None
        self.metadata = None
        self.extra_body = {}
        self.thinking = MagicMock()
        self.thinking.enabled = False


class AsyncStreamMock:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def test_init():
    config = ProviderConfig(
        api_key="test-maxplus-key",
        base_url="https://api.maxplus-ai.cc/v1",
    )
    with patch("providers.maxplus.client.AsyncOpenAI") as mock_client:
        provider = MaxPlusProvider(config)

    assert provider._api_key == "test-maxplus-key"
    assert provider._base_url == "https://api.maxplus-ai.cc/v1"
    mock_client.assert_called_once()


def test_build_request_body():
    config = ProviderConfig(
        api_key="test-maxplus-key",
        base_url="https://api.maxplus-ai.cc/v1",
    )
    with patch("providers.maxplus.client.AsyncOpenAI"):
        provider = MaxPlusProvider(config)

    body = provider._build_request_body(MockRequest())

    assert body["model"] == "gpt-5.4-mini"
    assert body["input"] == [{"role": "user", "content": "say pong"}]
    assert body["max_output_tokens"] == 32


def test_build_messages_request_body_for_claude():
    config = ProviderConfig(
        api_key="test-maxplus-key",
        base_url="https://api.maxplus-ai.cc/v1",
    )
    with patch("providers.maxplus.client.AsyncOpenAI"):
        provider = MaxPlusProvider(config)

    body = provider._build_messages_request_body(MockRequest("claude-sonnet-4-6"))

    assert body["model"] == "claude-sonnet-4-6"
    assert body["messages"] == [{"role": "user", "content": "say pong"}]
    assert body["max_tokens"] == 32


@pytest.mark.asyncio
async def test_list_model_ids_uses_static_catalog():
    config = ProviderConfig(
        api_key="test-maxplus-key",
        base_url="https://api.maxplus-ai.cc/v1",
    )
    with patch("providers.maxplus.client.AsyncOpenAI") as mock_client:
        provider = MaxPlusProvider(config)

    assert await provider.list_model_ids() == MAXPLUS_MODEL_IDS
    assert "claude-opus-4-7" in MAXPLUS_MODEL_IDS
    assert "claude-sonnet-4-6" in MAXPLUS_MODEL_IDS
    assert "gpt-5.4-mini" in MAXPLUS_MODEL_IDS
    mock_client.return_value.models.list.assert_not_called()


@pytest.mark.asyncio
async def test_stream_response_routes_claude_models_to_messages_api():
    config = ProviderConfig(
        api_key="test-maxplus-key",
        base_url="https://api.maxplus-ai.cc/v1",
    )
    with patch("providers.maxplus.client.AsyncOpenAI"):
        provider = MaxPlusProvider(config)

    response_calls = 0

    async def stream_messages(*args, **kwargs):
        yield "event: message_stop\n\n"

    async def stream_responses(*args, **kwargs):
        nonlocal response_calls
        response_calls += 1
        yield "data: responses\n\n"

    with (
        patch.object(
            MaxPlusProvider, "_stream_messages_response_impl", stream_messages
        ),
        patch.object(MaxPlusProvider, "_stream_response_impl", stream_responses),
    ):
        raw = "".join(
            [
                event
                async for event in provider.stream_response(
                    MockRequest("claude-opus-4-7")
                )
            ]
        )

    assert raw == "event: message_stop\n\n"
    assert response_calls == 0


@pytest.mark.asyncio
async def test_stream_response_routes_gpt_models_to_responses_api():
    config = ProviderConfig(
        api_key="test-maxplus-key",
        base_url="https://api.maxplus-ai.cc/v1",
    )
    with patch("providers.maxplus.client.AsyncOpenAI"):
        provider = MaxPlusProvider(config)

    messages_calls = 0

    async def stream_messages(*args, **kwargs):
        nonlocal messages_calls
        messages_calls += 1
        yield "event: message_stop\n\n"

    async def stream_responses(*args, **kwargs):
        yield "data: responses\n\n"

    with (
        patch.object(
            MaxPlusProvider, "_stream_messages_response_impl", stream_messages
        ),
        patch.object(MaxPlusProvider, "_stream_response_impl", stream_responses),
    ):
        raw = "".join(
            [event async for event in provider.stream_response(MockRequest())]
        )

    assert raw == "data: responses\n\n"
    assert messages_calls == 0


@pytest.mark.asyncio
async def test_stream_response_emits_tool_use_from_responses_function_call():
    config = ProviderConfig(
        api_key="test-maxplus-key",
        base_url="https://api.maxplus-ai.cc/v1",
    )
    with patch("providers.maxplus.client.AsyncOpenAI"):
        provider = MaxPlusProvider(config)

    function_item = SimpleNamespace(
        type="function_call",
        call_id="call_1",
        id="item_1",
        name="Read",
        arguments="",
    )
    events = [
        SimpleNamespace(
            type="response.output_item.added",
            output_index=0,
            item=function_item,
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            output_index=0,
            delta='{"file_path":',
        ),
        SimpleNamespace(
            type="response.function_call_arguments.delta",
            output_index=0,
            delta='"README.md"}',
        ),
        SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(status="completed"),
        ),
    ]

    async def execute_with_retry(*args, **kwargs):
        return AsyncStreamMock(events)

    with patch.object(
        provider._global_rate_limiter,
        "execute_with_retry",
        execute_with_retry,
    ):
        raw = "".join(
            [event async for event in provider.stream_response(MockRequest())]
        )

    assert '"type": "tool_use"' in raw
    assert '"id": "call_1"' in raw
    assert '"name": "Read"' in raw
    assert '"partial_json": "{\\"file_path\\":"' in raw
    assert '"partial_json": "\\"README.md\\"}"' in raw
    assert '"stop_reason": "tool_use"' in raw
