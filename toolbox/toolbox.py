import inspect
import json
import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessage

from toolbox.messages import (
    ChatMessage,
    ErrorResult,
    Result,
    SuccessResult,
    serialize_results,
)
from toolbox.schema import (
    Function,
    Parameter,
    build_tools_schema,
    python_type_to_json_schema_type,
)

logger = logging.getLogger(__name__)


class Toolbox:
    def __init__(self):
        # Maps function names to Function data objects (may be partially populated)
        self._functions_data: dict[str, Function] = {}

    @property
    def tools(self):
        """Returns the list of tool definitions for the OpenAI API."""
        return build_tools_schema(self._functions_data.values())

    def parameter(
        self,
        name: str,
        type: str | None = None,
        description: str | None = None,
        required: bool = True,
        enum: list[str | int | float | bool] | None = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to define a parameter for the tool.
        Since decorators run bottom-to-top, these attach metadata to the function
        before @toolbox.function registers it.

        Args:
            name: The parameter name
            type: Optional JSON schema type (e.g., "string", "integer"). If not provided,
                  will be extracted from the function's type annotation.
            description: Description of the parameter
            required: Whether the parameter is required
            enum: Optional list of allowed values
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func_name = func.__name__

            # If type is not provided, extract it from the function's type annotation
            param_type = type
            if param_type is None:
                sig = inspect.signature(func)
                param = sig.parameters.get(name)
                if param is None:
                    raise ValueError(
                        f"Parameter '{name}' not found in function '{func.__name__}' signature"
                    )

                annotation = param.annotation
                if annotation == inspect.Parameter.empty:
                    raise ValueError(
                        f"Parameter '{name}' in function '{func.__name__}' has no type annotation "
                        + "and no 'type' argument was provided. Either provide a type annotation "
                        + "or specify the 'type' argument in the decorator."
                    )

                param_type = python_type_to_json_schema_type(annotation)

            # Create Parameter data object
            param_obj = Parameter(
                name=name,
                type=param_type,
                description=description,
                required=required,
                enum=enum,
            )

            # Get or create Function object for this function name
            if func_name not in self._functions_data:
                self._functions_data[func_name] = Function()

            # Append Parameter to the Function's parameters list
            self._functions_data[func_name].parameters.append(param_obj)

            # Return function unchanged (no modification to function object)
            return func

        return decorator

    def function(
        self, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register the function as a tool.
        This must be placed *above* @toolbox.parameter decorators.

        Args:
            description: Optional description of the function. If not provided,
                        the function's docstring will be used.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func_name = func.__name__

            # Get or create Function object from _functions_data
            if func_name not in self._functions_data:
                self._functions_data[func_name] = Function()

            # Use provided description or fall back to function's docstring
            func_description = description
            if func_description is None:
                func_description = inspect.getdoc(func) or ""

            # Populate/update Function object with name, description, and callable
            func_data = self._functions_data[func_name]
            func_data.name = func_name
            func_data.description = func_description
            func_data.callable = func

            # Return function unchanged (or wrapped if needed for execution)
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def execute(
        self, message: "ChatCompletionMessage"
    ) -> tuple[ChatMessage, list[ChatMessage]]:
        """
        Executes tool calls from an OpenAI chat completion message.

        Args:
            message: The ChatCompletionMessage containing tool_calls

        Returns:
            Tuple of (assistant_message_dict, list_of_tool_result_dicts):
            - Assistant message with tool_calls (serialized)
            - Tool response messages
        """
        tool_calls = message.tool_calls or []

        results: list[Result] = []

        for tool_call in tool_calls:
            # Only handle function tool calls (not custom tool calls)
            if tool_call.type != "function":
                logger.warning(
                    f"Skipping tool call of type '{tool_call.type}' - only 'function' type is supported"
                )
                continue

            fn_name = tool_call.function.name
            fn_args_str = tool_call.function.arguments

            if fn_name not in self._functions_data:
                print(f"Error: Function {fn_name} not found in toolbox.")
                continue

            func_data = self._functions_data[fn_name]
            if func_data.callable is None:
                print(f"Error: Function {fn_name} has no callable.")
                continue

            try:
                # Parse JSON arguments from LLM
                fn_args = json.loads(fn_args_str)
                logger.info(
                    f"Invoking tool: {fn_name}({', '.join(f'{k}={v}' for k, v in fn_args.items())})"
                )

                # Execute the actual Python function
                output = func_data.callable(**fn_args)

                # Store result as SuccessResult dataclass
                results.append(
                    SuccessResult(
                        tool_call=tool_call,
                        name=fn_name,
                        output=output,
                    )
                )

            except Exception as e:
                # Store result as ErrorResult dataclass
                results.append(
                    ErrorResult(
                        tool_call=tool_call,
                        name=fn_name,
                        error=e,
                    )
                )

        # Serialize assistant message and results at the end
        return serialize_results(results)
