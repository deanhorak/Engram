"""Interactive chat loop for the optimized native BitNet package runtime."""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

from engram.runtime.native_bitnet import NativeBitNetRuntime


def run_native_bitnet_chat(
    runtime: NativeBitNetRuntime,
    *,
    max_new_tokens: int = 32,
    system_prompt: str | None = "You are a helpful assistant.",
    attention_library=None,
    local_window: int = 16,
    older_candidates: int = 8,
    older_top_k: int = 4,
    sink_tokens: int = 2,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
    on_turn: Callable[[dict], None] | None = None,
) -> int:
    """Run a re-prefilled chat session and return completed assistant turns."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    template = getattr(runtime.tokenizer, "chat_template", None)
    if not template:
        raise ValueError("the packaged tokenizer does not define a chat template")
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    initial_messages = list(messages)
    turns = 0
    output_stream.write(
        "Engram native BitNet chat. Commands: /reset, /history, /quit\n"
    )
    output_stream.flush()

    while True:
        output_stream.write("You> ")
        output_stream.flush()
        line = input_stream.readline()
        if line == "":
            output_stream.write("\n")
            break
        user_text = line.strip()
        if not user_text:
            continue
        command = user_text.lower()
        if command in {"/quit", "/exit"}:
            break
        if command == "/reset":
            messages = list(initial_messages)
            output_stream.write("Conversation reset.\n")
            continue
        if command == "/history":
            conversational = [
                item for item in messages if item["role"] != "system"
            ]
            if not conversational:
                output_stream.write("(empty)\n")
            for item in conversational:
                output_stream.write(
                    f"{item['role'].capitalize()}: {item['content']}\n"
                )
            continue
        if command.startswith("/"):
            output_stream.write(f"Unknown command: {user_text}\n")
            continue

        messages.append({"role": "user", "content": user_text})
        rendered = runtime.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        result = runtime.generate_bounded(
            rendered,
            max_new_tokens=max_new_tokens,
            attention_library=attention_library,
            local_window=local_window,
            older_candidates=older_candidates,
            older_top_k=older_top_k,
            sink_tokens=sink_tokens,
        )
        assistant_text = result.text.strip()
        messages.append({"role": "assistant", "content": assistant_text})
        turns += 1
        output_stream.write(f"Engram> {assistant_text}\n")
        output_stream.write(
            f"[{result.elapsed_seconds:.2f}s; "
            f"{len(result.generated_tokens)} tokens; "
            f"{result.attention_state_bytes} attention-state bytes]\n"
        )
        output_stream.flush()
        if on_turn is not None:
            on_turn(
                {
                    "turn": turns,
                    "messages": [dict(item) for item in messages],
                    "rendered_prompt": rendered,
                    "result": result,
                }
            )
    return turns


__all__ = ["run_native_bitnet_chat"]
