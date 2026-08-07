from flab.data import format_messages


def test_format_messages_renders_roles_in_order():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = format_messages(msgs)
    assert out == "<|user|>\nhi\n<|assistant|>\nhello\n<|end|>"


def test_format_messages_supports_system_role():
    msgs = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
    ]
    out = format_messages(msgs)
    assert out.startswith("<|system|>\nbe brief\n<|user|>")
