#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


HELP = """commands:
  /help          show this help
  /reset         clear conversation history
  /usage         toggle token usage after replies
  /exit, /quit   leave chat
"""


def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def chat_once(
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        timeout: float) -> tuple[str, dict]:
    result = post_json(
        base_url.rstrip("/") + "/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        timeout,
    )
    choices = result.get("choices") or []
    if not choices:
        raise RuntimeError("response has no choices")
    message = choices[0].get("message") or {}
    return str(message.get("content") or ""), result


def print_usage(result: dict) -> None:
    usage = result.get("usage") or {}
    if not usage:
        return
    print(
        "[usage] "
        f"prompt={usage.get('prompt_tokens', 0)} "
        f"completion={usage.get('completion_tokens', 0)} "
        f"total={usage.get('total_tokens', 0)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chat with Sandy's OpenAI-compatible HTTP server."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--system", default=None)
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument("--show-usage", action="store_true")
    args = parser.parse_args()

    if args.max_tokens < 0:
        print("--max-tokens must be >= 0", file=sys.stderr)
        return 1

    base_url = args.base_url.rstrip("/")
    model = args.model
    if model is None:
        try:
            models = get_json(base_url + "/v1/models", args.timeout)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"failed to fetch model list: {exc}", file=sys.stderr)
            return 1
        data = models.get("data") or []
        if not data or "id" not in data[0]:
            print("server returned no model id", file=sys.stderr)
            return 1
        model = str(data[0]["id"])

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    show_usage = args.show_usage

    def ask(prompt: str) -> int:
        request_messages = messages + [{"role": "user", "content": prompt}]
        try:
            answer, result = chat_once(
                base_url,
                model,
                request_messages,
                args.max_tokens,
                args.timeout,
            )
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as exc:
            print(f"request failed: {exc}", file=sys.stderr)
            return 1

        print(f"assistant> {answer}")
        if show_usage:
            print_usage(result)
        if not args.no_history:
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": answer})
        return 0

    print(f"[chat] {base_url} model={model}")
    print("[chat] type /help for commands")
    while True:
        try:
            prompt = input("you> ")
        except EOFError:
            print()
            return 0
        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt == "/help":
            print(HELP, end="")
            continue
        if prompt in {"/exit", "/quit"}:
            return 0
        if prompt == "/reset":
            messages.clear()
            if args.system:
                messages.append({"role": "system", "content": args.system})
            print("[chat] history cleared")
            continue
        if prompt == "/usage":
            show_usage = not show_usage
            print(f"[chat] usage {'on' if show_usage else 'off'}")
            continue
        code = ask(prompt)
        if code != 0:
            return code


if __name__ == "__main__":
    raise SystemExit(main())
