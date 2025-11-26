from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageFunctionToolCall

# Type alias for chat messages
ChatMessage = dict[str, Any]


@dataclass
class SuccessResult:
    """Represents a successful tool execution result."""

    tool_call: "ChatCompletionMessageFunctionToolCall"
    name: str
    output: Any

    @property
    def content(self):
        return str(self.output)


@dataclass
class ErrorResult:
    """Represents an error during tool execution."""

    tool_call: "ChatCompletionMessageFunctionToolCall"
    name: str
    error: Exception

    @property
    def content(self):
        return f"Error executing {self.name}: {self.error}"


Result = SuccessResult | ErrorResult


def serialize_results(results: list[Result]):
    """
    Converts a list of Result dataclasses to a list of dicts.

    Args:
        results: List of Result dataclasses (SuccessResult or ErrorResult)

    Returns:
        List of dicts representing the results
    """
    assistant_message: ChatMessage = {
        "role": "assistant",
        "tool_calls": [],
    }
    serialized_results: list[ChatMessage] = []
    for result in results:
        assistant_message["tool_calls"].append(result.tool_call.model_dump())
        serialized_results.append(
            {
                "tool_call_id": result.tool_call.id,
                "role": "tool",
                "name": result.name,
                "content": result.content,
            }
        )
    return assistant_message, serialized_results
