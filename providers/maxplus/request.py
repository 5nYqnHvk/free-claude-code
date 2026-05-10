"""Request builder for MaxPlus provider."""

from typing import Any

from loguru import logger

from core.anthropic import ReasoningReplayMode, build_base_request_body
from core.anthropic.conversion import OpenAIConversionError
from providers.exceptions import InvalidRequestError

_RESPONSES_ONLY_KEYS = {
    "model",
    "input",
    "instructions",
    "max_output_tokens",
    "temperature",
    "top_p",
    "tools",
    "tool_choice",
    "metadata",
}


def _responses_input_from_openai_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    input_items: list[dict[str, Any]] = []
    instructions: list[str] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            if content:
                instructions.append(str(content))
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id", ""),
                    "output": str(content),
                }
            )
            continue
        if role == "assistant" and message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                input_items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "arguments": function.get("arguments", "{}"),
                    }
                )
            if not content:
                continue
        input_items.append({"role": role, "content": content})
    return input_items, "\n\n".join(instructions) if instructions else None


def _responses_tools_from_openai_tools(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function", {})
        converted.append(
            {
                "type": "function",
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )
    return converted


def _responses_tool_choice_from_openai(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    choice_type = tool_choice.get("type")
    if choice_type == "function" and isinstance(tool_choice.get("function"), dict):
        name = tool_choice["function"].get("name")
        if name:
            return {"type": "function", "name": name}
    return tool_choice


def build_request_body(request_data: Any, *, thinking_enabled: bool) -> dict[str, Any]:
    """Build OpenAI Responses API request body from Anthropic request."""
    logger.debug(
        "MAXPLUS_REQUEST: conversion start model={} msgs={}",
        getattr(request_data, "model", "?"),
        len(getattr(request_data, "messages", [])),
    )
    try:
        openai_body = build_base_request_body(
            request_data,
            reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
        )
    except OpenAIConversionError as exc:
        raise InvalidRequestError(str(exc)) from exc

    input_items, instructions = _responses_input_from_openai_messages(
        openai_body.get("messages", [])
    )
    body: dict[str, Any] = {
        "model": openai_body["model"],
        "input": input_items,
    }
    if instructions:
        body["instructions"] = instructions
    if max_tokens := openai_body.get("max_tokens"):
        body["max_output_tokens"] = max_tokens
    if temperature := openai_body.get("temperature"):
        body["temperature"] = temperature
    if top_p := openai_body.get("top_p"):
        body["top_p"] = top_p
    if tools := openai_body.get("tools"):
        body["tools"] = _responses_tools_from_openai_tools(tools)
    if tool_choice := openai_body.get("tool_choice"):
        body["tool_choice"] = _responses_tool_choice_from_openai(tool_choice)
    if metadata := getattr(request_data, "metadata", None):
        body["metadata"] = metadata

    extra_body = getattr(request_data, "extra_body", None) or {}
    if extra_body:
        reserved = sorted(set(extra_body) & _RESPONSES_ONLY_KEYS)
        if reserved:
            raise InvalidRequestError(
                f"MaxPlus extra_body cannot override reserved fields: {reserved}"
            )
        body.update(extra_body)

    logger.debug(
        "MAXPLUS_REQUEST: conversion done model={} input_items={} tools={}",
        body.get("model"),
        len(body.get("input", [])),
        len(body.get("tools", [])),
    )
    return body
