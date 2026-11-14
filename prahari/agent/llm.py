"""LLM adapter interface with three providers: none, anthropic, openai_compatible.

Messages use one neutral format: {"role": "user"|"assistant"|"tool", ...}; assistant
messages carry `text` and `tool_calls` ([{id, name, args}]), tool messages carry
`tool_call_id`, `name`, `content`. Each provider converts to its wire format.
HTTP goes through an injectable `transport` so tests never touch the network.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

Transport = Callable[[str, dict, dict], dict]  # (url, headers, json_body) -> json


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    """One model turn: free text and/or tool calls."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


def http_transport(url: str, headers: dict, body: dict) -> dict:
    """POST JSON with the standard library; raises on HTTP errors."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"content-type": "application/json", **headers}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - https base url from env
        return json.loads(resp.read())


class LLMProvider(ABC):
    """Interface every provider implements."""

    name: str = "base"
    model: str = ""
    uses_tools: bool = True

    @abstractmethod
    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        """Run one turn."""


class NoneProvider(LLMProvider):
    """No LLM: the agent uses its deterministic plan and template narrative."""

    name = "none"
    model = "template-v1"
    uses_tools = False

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        raise RuntimeError("the 'none' provider has no chat; use the deterministic plan")


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API (tool_use blocks)."""

    name = "anthropic"

    def __init__(
        self,
        model: str = "",
        api_key: str | None = None,
        transport: Transport | None = None,
        base_url: str = "https://api.anthropic.com",
    ):
        self.model = model or os.environ.get("PRAHARI_LLM_MODEL", "claude-sonnet-4-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.transport = transport or http_transport

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        wire: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks: list[dict] = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                blocks += [
                    {"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                    for c in m.get("tool_calls", [])
                ]
                wire.append({"role": "assistant", "content": blocks})
            else:
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }
                if wire and wire[-1]["role"] == "user" and isinstance(wire[-1]["content"], list):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
        body = {
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": wire,
            "tools": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["parameters"],
                }
                for t in tools
            ],
        }
        data = self.transport(
            f"{self.base_url}/v1/messages",
            {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
            body,
        )
        out = LLMResponse()
        for b in data.get("content", []):
            if b.get("type") == "text":
                out.text += b.get("text", "")
            elif b.get("type") == "tool_use":
                out.tool_calls.append(ToolCall(b["id"], b["name"], b.get("input", {})))
        return out


class OpenAICompatibleProvider(LLMProvider):
    """Any OpenAI-style /chat/completions endpoint (base URL and key from env)."""

    name = "openai_compatible"

    def __init__(
        self,
        model: str = "",
        base_url: str | None = None,
        api_key: str | None = None,
        transport: Transport | None = None,
    ):
        self.model = model or os.environ.get("PRAHARI_LLM_MODEL", "")
        self.base_url = (base_url or os.environ.get("PRAHARI_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("PRAHARI_LLM_API_KEY", "")
        self.transport = transport or http_transport

    def chat(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        wire: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                wire.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                msg: dict = {"role": "assistant", "content": m.get("text") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": c["id"],
                            "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["args"])},
                        }
                        for c in m["tool_calls"]
                    ]
                wire.append(msg)
            else:
                wire.append(
                    {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]}
                )
        body = {
            "model": self.model,
            "messages": wire,
            "tools": [{"type": "function", "function": t} for t in tools],
        }
        data = self.transport(
            f"{self.base_url}/chat/completions", {"authorization": f"Bearer {self.api_key}"}, body
        )
        msg = data["choices"][0]["message"]
        out = LLMResponse(text=msg.get("content") or "")
        for c in msg.get("tool_calls") or []:
            out.tool_calls.append(
                ToolCall(
                    c["id"],
                    c["function"]["name"],
                    json.loads(c["function"].get("arguments") or "{}"),
                )
            )
        return out


def make_provider(name: str, model: str = "", **kw) -> LLMProvider:
    """Build a provider by name; `stub` is accepted as an alias of `none`."""
    key = (name or "none").lower()
    if key in ("none", "stub"):
        return NoneProvider()
    if key == "anthropic":
        return AnthropicProvider(model, **kw)
    if key in ("openai_compatible", "openai"):
        return OpenAICompatibleProvider(model, **kw)
    raise ValueError(f"unknown llm provider: {name}")
