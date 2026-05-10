"""MaxPlus provider implementation."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from config.constants import ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS
from core.anthropic import SSEBuilder, iter_provider_stream_error_sse_events
from core.anthropic.emitted_sse_tracker import EmittedNativeSseTracker
from core.anthropic.native_messages_request import (
    build_base_native_anthropic_request_body,
)
from providers.base import BaseProvider, ProviderConfig
from providers.defaults import MAXPLUS_DEFAULT_BASE
from providers.error_mapping import (
    map_error,
    user_visible_message_for_mapped_provider_error,
)
from providers.rate_limit import GlobalRateLimiter

from .request import build_request_body

MAXPLUS_MODEL_IDS = frozenset(
    {
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-haiku-4-5-20251001",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    }
)


class MaxPlusProvider(BaseProvider):
    """MaxPlus provider using Anthropic Messages for Claude and Responses for GPT."""

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._api_key = config.api_key
        self._base_url = (config.base_url or MAXPLUS_DEFAULT_BASE).rstrip("/")
        self._global_rate_limiter = GlobalRateLimiter.get_scoped_instance(
            "maxplus",
            rate_limit=config.rate_limit,
            rate_window=config.rate_window,
            max_concurrency=config.max_concurrency,
        )
        http_client = None
        if config.proxy:
            http_client = httpx.AsyncClient(
                proxy=config.proxy,
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    read=config.http_read_timeout,
                    write=config.http_write_timeout,
                ),
            )
        timeout = httpx.Timeout(
            config.http_read_timeout,
            connect=config.http_connect_timeout,
            read=config.http_read_timeout,
            write=config.http_write_timeout,
        )
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            max_retries=0,
            timeout=timeout,
            http_client=http_client,
        )
        self._messages_client = httpx.AsyncClient(
            base_url=self._base_url,
            proxy=config.proxy or None,
            timeout=timeout,
        )

    async def cleanup(self) -> None:
        """Release HTTP client resources."""
        client = getattr(self, "_client", None)
        if client is not None:
            await client.aclose()
        await self._messages_client.aclose()

    async def list_model_ids(self) -> frozenset[str]:
        """Return MaxPlus model ids without querying an unsupported model-list path."""
        return MAXPLUS_MODEL_IDS

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict[str, Any]:
        return build_request_body(
            request,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _build_messages_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict[str, Any]:
        return build_base_native_anthropic_request_body(
            request,
            default_max_tokens=ANTHROPIC_DEFAULT_MAX_OUTPUT_TOKENS,
            thinking_enabled=self._is_thinking_enabled(request, thinking_enabled),
        )

    def _uses_messages_api(self, model: str) -> bool:
        return model.startswith("claude-")

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        thinking_enabled: bool | None = None,
    ) -> AsyncIterator[str]:
        """Stream MaxPlus output in Anthropic SSE format."""
        with logger.contextualize(request_id=request_id):
            if self._uses_messages_api(request.model):
                async for event in self._stream_messages_response_impl(
                    request,
                    input_tokens,
                    request_id,
                    thinking_enabled=thinking_enabled,
                ):
                    yield event
                return
            async for event in self._stream_response_impl(
                request,
                input_tokens,
                request_id,
                thinking_enabled=thinking_enabled,
            ):
                yield event

    async def _stream_messages_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
    ) -> AsyncIterator[str]:
        body = self._build_messages_request_body(
            request, thinking_enabled=thinking_enabled
        )
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            "MAXPLUS_STREAM:{} natively passing Anthropic request model={} msgs={} tools={}",
            req_tag,
            body.get("model"),
            len(body.get("messages", [])),
            len(body.get("tools", [])),
        )

        response: httpx.Response | None = None
        sent_any_event = False
        emitted_tracker = EmittedNativeSseTracker()
        async with self._global_rate_limiter.concurrency_slot():
            try:

                async def _validated_stream_send() -> httpx.Response:
                    send_response = await self._messages_client.send(
                        self._messages_client.build_request(
                            "POST",
                            "/messages",
                            json=body,
                            headers={
                                "Accept": "text/event-stream",
                                "Content-Type": "application/json",
                                "x-api-key": self._api_key,
                            },
                        ),
                        stream=True,
                    )
                    if send_response.status_code != 200:
                        try:
                            send_response.raise_for_status()
                        finally:
                            if not send_response.is_closed:
                                await send_response.aclose()
                    return send_response

                response = await self._global_rate_limiter.execute_with_retry(
                    _validated_stream_send
                )
                async for line in response.aiter_lines():
                    chunk = f"{line}\n" if line else "\n"
                    sent_any_event = True
                    emitted_tracker.feed(chunk)
                    yield chunk
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_stream_transport_error("MAXPLUS", req_tag, exc)
                mapped = map_error(exc, rate_limiter=self._global_rate_limiter)
                error_message = user_visible_message_for_mapped_provider_error(
                    mapped,
                    provider_name="MAXPLUS",
                    read_timeout_s=self._config.http_read_timeout,
                )
                if response is not None and not response.is_closed:
                    await response.aclose()
                if sent_any_event:
                    for event in emitted_tracker.iter_close_unclosed_blocks():
                        yield event
                    for event in emitted_tracker.iter_midstream_error_tail(
                        error_message,
                        request=request,
                        input_tokens=input_tokens,
                        log_raw_sse_events=self._config.log_raw_sse_events,
                    ):
                        yield event
                else:
                    for event in iter_provider_stream_error_sse_events(
                        request=request,
                        input_tokens=input_tokens,
                        error_message=error_message,
                        sent_any_event=False,
                        log_raw_sse_events=self._config.log_raw_sse_events,
                    ):
                        yield event
                return
            finally:
                if response is not None and not response.is_closed:
                    await response.aclose()

    def _start_tool_block(
        self,
        sse: SSEBuilder,
        output_index: int,
        tool_id: str,
        name: str,
    ) -> Iterator[str]:
        yield from sse.close_content_blocks()
        yield sse.start_tool_block(output_index, tool_id, name or "tool_call")

    async def _stream_response_impl(
        self,
        request: Any,
        input_tokens: int,
        request_id: str | None,
        *,
        thinking_enabled: bool | None,
    ) -> AsyncIterator[str]:
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(
            message_id,
            request.model,
            input_tokens,
            log_raw_events=self._config.log_raw_sse_events,
        )
        body = self._build_request_body(request, thinking_enabled=thinking_enabled)
        req_tag = f" request_id={request_id}" if request_id else ""
        body_input = body.get("input", [])
        input_count = len(body_input) if isinstance(body_input, list) else 1
        logger.info(
            "MAXPLUS_STREAM:{} model={} input_items={}",
            req_tag,
            body.get("model"),
            input_count,
        )

        yield sse.message_start()
        finish_reason = "end_turn"
        tool_ids: dict[int, str] = {}
        tool_names: dict[int, str] = {}
        tool_started: set[int] = set()
        tool_args_emitted: set[int] = set()
        has_tool_use = False

        async with self._global_rate_limiter.concurrency_slot():
            try:
                stream = await self._global_rate_limiter.execute_with_retry(
                    self._client.responses.create,
                    **body,
                    stream=True,
                )
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        if delta:
                            for item in sse.ensure_text_block():
                                yield item
                            yield sse.emit_text_delta(delta)
                    elif event_type == "response.output_item.added":
                        item = getattr(event, "item", None)
                        if getattr(item, "type", "") == "function_call":
                            output_index = int(getattr(event, "output_index", 0) or 0)
                            tool_id = str(
                                getattr(item, "call_id", None)
                                or getattr(item, "id", None)
                                or f"tool_{uuid.uuid4()}"
                            )
                            name = str(getattr(item, "name", "") or "tool_call")
                            tool_ids[output_index] = tool_id
                            tool_names[output_index] = name
                            if output_index not in tool_started:
                                for item_event in self._start_tool_block(
                                    sse,
                                    output_index,
                                    tool_id,
                                    name,
                                ):
                                    yield item_event
                                tool_started.add(output_index)
                                has_tool_use = True
                            arguments = getattr(item, "arguments", "") or ""
                            if arguments:
                                yield sse.emit_tool_delta(output_index, arguments)
                                tool_args_emitted.add(output_index)
                    elif event_type == "response.function_call_arguments.delta":
                        output_index = int(getattr(event, "output_index", 0) or 0)
                        delta = getattr(event, "delta", "") or ""
                        if delta and output_index in tool_started:
                            yield sse.emit_tool_delta(output_index, delta)
                            tool_args_emitted.add(output_index)
                    elif event_type == "response.function_call_arguments.done":
                        output_index = int(getattr(event, "output_index", 0) or 0)
                        if output_index not in tool_started:
                            tool_id = str(
                                tool_ids.get(output_index)
                                or getattr(event, "item_id", None)
                                or f"tool_{uuid.uuid4()}"
                            )
                            name = str(
                                getattr(event, "name", None)
                                or tool_names.get(output_index)
                                or "tool_call"
                            )
                            for item_event in self._start_tool_block(
                                sse,
                                output_index,
                                tool_id,
                                name,
                            ):
                                yield item_event
                            tool_started.add(output_index)
                            has_tool_use = True
                        arguments = getattr(event, "arguments", "") or ""
                        if arguments and output_index not in tool_args_emitted:
                            yield sse.emit_tool_delta(output_index, arguments)
                            tool_args_emitted.add(output_index)
                    elif event_type == "response.output_item.done":
                        item = getattr(event, "item", None)
                        if getattr(item, "type", "") == "function_call":
                            output_index = int(getattr(event, "output_index", 0) or 0)
                            if output_index not in tool_started:
                                tool_id = str(
                                    getattr(item, "call_id", None)
                                    or getattr(item, "id", None)
                                    or f"tool_{uuid.uuid4()}"
                                )
                                name = str(getattr(item, "name", "") or "tool_call")
                                for item_event in self._start_tool_block(
                                    sse,
                                    output_index,
                                    tool_id,
                                    name,
                                ):
                                    yield item_event
                                tool_started.add(output_index)
                                has_tool_use = True
                            arguments = getattr(item, "arguments", "") or ""
                            if arguments and output_index not in tool_args_emitted:
                                yield sse.emit_tool_delta(output_index, arguments)
                                tool_args_emitted.add(output_index)
                    elif event_type == "response.completed":
                        response = getattr(event, "response", None)
                        status = getattr(response, "status", None)
                        if status == "incomplete":
                            finish_reason = "max_tokens"
                    elif event_type == "response.incomplete":
                        finish_reason = "max_tokens"

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._log_stream_transport_error("MAXPLUS", req_tag, exc)
                mapped = map_error(exc, rate_limiter=self._global_rate_limiter)
                error_message = user_visible_message_for_mapped_provider_error(
                    mapped,
                    provider_name="MAXPLUS",
                    read_timeout_s=self._config.http_read_timeout,
                )
                for item in sse.close_all_blocks():
                    yield item
                for item in sse.emit_error(error_message):
                    yield item
                yield sse.message_delta("end_turn", 1)
                yield sse.message_stop()
                return

        if not has_tool_use and not sse.accumulated_text.strip():
            for item in sse.ensure_text_block():
                yield item
            yield sse.emit_text_delta(" ")

        for item in sse.close_all_blocks():
            yield item
        yield sse.message_delta(
            "tool_use" if has_tool_use else finish_reason,
            sse.estimate_output_tokens(),
        )
        yield sse.message_stop()
