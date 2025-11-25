from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union

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


@dataclass
class ErrorResult:
    """Represents an error during tool execution."""

    tool_call: "ChatCompletionMessageFunctionToolCall"
    name: str
    error: Exception


Result = Union[SuccessResult, ErrorResult]


def serialize_results(
    results: list[Result],
) -> tuple[ChatMessage, list[ChatMessage]]:
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
        if isinstance(result, SuccessResult):
            content = str(result.output)
        elif isinstance(result, ErrorResult):
            content = f"Error executing {result.name}: {result.error}"
        else:
            raise TypeError(f"Unknown result type: {type(result)}")

        assistant_message["tool_calls"].append(result.tool_call.model_dump())
        serialized_results.append(
            {
                "tool_call_id": result.tool_call.id,
                "role": "tool",
                "name": result.name,
                "content": content,
            }
        )
    return assistant_message, serialized_results
