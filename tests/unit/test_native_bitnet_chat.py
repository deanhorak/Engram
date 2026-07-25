from io import StringIO
from types import SimpleNamespace

import pytest

from engram.runtime.native_bitnet_chat import run_native_bitnet_chat


class _Tokenizer:
    chat_template = "fixture"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        copied = [dict(item) for item in messages]
        self.calls.append(copied)
        return "|".join(f"{item['role']}:{item['content']}" for item in copied)


class _Runtime:
    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.prompts = []

    def generate_bounded(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        turn = len(self.prompts)
        return SimpleNamespace(
            text=f"answer {turn}",
            elapsed_seconds=float(turn),
            generated_tokens=(turn, turn + 1),
            attention_state_bytes=1234,
        )


def test_chat_uses_template_and_represents_complete_history_each_turn():
    runtime = _Runtime()
    output = StringIO()
    turns = run_native_bitnet_chat(
        runtime,
        max_new_tokens=7,
        system_prompt="system text",
        attention_library="attention.so",
        input_stream=StringIO("hello\nfollow up\n/quit\n"),
        output_stream=output,
    )

    assert turns == 2
    assert runtime.tokenizer.calls[0] == [
        {"role": "system", "content": "system text"},
        {"role": "user", "content": "hello"},
    ]
    assert runtime.tokenizer.calls[1][-2:] == [
        {"role": "assistant", "content": "answer 1"},
        {"role": "user", "content": "follow up"},
    ]
    assert runtime.prompts[0][1]["max_new_tokens"] == 7
    assert runtime.prompts[0][1]["attention_library"] == "attention.so"
    assert "Engram> answer 1" in output.getvalue()


def test_chat_reset_history_unknown_command_and_eof():
    runtime = _Runtime()
    output = StringIO()
    turns = run_native_bitnet_chat(
        runtime,
        system_prompt=None,
        input_stream=StringIO("first\n/reset\n/history\n/nope\nsecond\n"),
        output_stream=output,
    )

    assert turns == 2
    assert runtime.tokenizer.calls[1] == [
        {"role": "user", "content": "second"}
    ]
    text = output.getvalue()
    assert "Conversation reset." in text
    assert "(empty)" in text
    assert "Unknown command: /nope" in text


def test_chat_rejects_tokenizer_without_template():
    runtime = _Runtime()
    runtime.tokenizer.chat_template = None
    with pytest.raises(ValueError, match="chat template"):
        run_native_bitnet_chat(
            runtime,
            input_stream=StringIO("/quit\n"),
            output_stream=StringIO(),
        )
