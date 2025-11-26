# pyright: reportUnusedFunction=false
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call import Function

from toolbox import Toolbox


def test_hello_world():
    toolbox = Toolbox()

    @toolbox.function(description="A hello world function that greets someone")
    @toolbox.parameter(
        name="who", type="string", description="The name of the person to greet"
    )
    def hello_world(who: str):
        return f"Hello {who}"

    assert toolbox.tools == [
        {
            "type": "function",
            "function": {
                "name": "hello_world",
                "description": "A hello world function that greets someone",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "who": {
                            "type": "string",
                            "description": "The name of the person to greet",
                        }
                    },
                    "required": ["who"],
                },
            },
        }
    ]

    message = ChatCompletionMessage(
        role="assistant",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="123",
                type="function",
                function=Function(
                    name="hello_world",
                    arguments='{"who": "world"}',
                ),
            ),
        ],
    )

    assistant_message, tool_results = toolbox.execute(message)
    assert assistant_message == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "123",
                "type": "function",
                "function": {
                    "name": "hello_world",
                    "arguments": '{"who": "world"}',
                },
            },
        ],
    }
    assert tool_results == [
        {
            "tool_call_id": "123",
            "role": "tool",
            "name": "hello_world",
            "content": "Hello world",
        },
    ]
