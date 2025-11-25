import inspect
import json
import logging
from functools import wraps
from numbers import Number
from typing import Any, Callable, Optional, Union, get_args, get_origin

logger = logging.getLogger(__name__)


def _python_type_to_json_schema_type(python_type: Any) -> str:
    """
    Converts Python type annotations to JSON schema type strings.
    """
    # Handle typing.Optional and Union types
    origin = get_origin(python_type)
    if origin is Union or origin is Optional:
        args = get_args(python_type)
        # For Optional[T], get the non-None type
        non_none_args = [arg for arg in args if arg is not type(None)]
        if non_none_args:
            return _python_type_to_json_schema_type(non_none_args[0])

    # Handle basic types
    type_mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        Number: "number",
    }

    # Check if it's a direct type match
    if python_type in type_mapping:
        return type_mapping[python_type]

    # Handle string annotations (forward references)
    if isinstance(python_type, str):
        type_mapping_str = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        return type_mapping_str.get(python_type, "string")

    # Default to string for unknown types
    return "string"


class Toolbox:
    def __init__(self) -> None:
        # Maps function names to the actual python callable
        self._functions: dict[str, Callable[..., Any]] = {}
        self._tools_schema: list[dict[str, Any]] = []

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Returns the list of tool definitions for the OpenAI API."""
        return self._tools_schema

    def parameter(
        self,
        name: str,
        type: Optional[str] = None,
        description: Optional[str] = None,
        required: bool = True,
        enum: Optional[list[Any]] = None,
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
            if not hasattr(func, "_tool_params"):
                func._tool_params = {}  # type: ignore[attr-defined]

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
                        "and no 'type' argument was provided. Either provide a type annotation "
                        "or specify the 'type' argument in the decorator."
                    )

                param_type = _python_type_to_json_schema_type(annotation)

            func._tool_params[name] = {"type": param_type, "description": description}  # type: ignore[attr-defined]

            if enum:
                func._tool_params[name]["enum"] = enum  # type: ignore[attr-defined]

            if not hasattr(func, "_tool_required"):
                func._tool_required = []  # type: ignore[attr-defined]

            if required:
                func._tool_required.append(name)  # type: ignore[attr-defined]

            return func

        return decorator

    def function(
        self, description: str
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """
        Decorator to register the function as a tool.
        This must be placed *above* @toolbox.parameter decorators.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            func_name = func.__name__

            # Get parameters defined by @parameter decorators
            properties = getattr(func, "_tool_params", {})
            required = getattr(func, "_tool_required", [])

            # Construct the OpenAI Tool Schema
            tool_definition = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": description,
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }

            # Register schema and callable
            self._tools_schema.append(tool_definition)
            self._functions[func_name] = func

            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def execute(self, tool_calls: Any) -> list[dict[str, Any]]:
        """
        Parses OpenAI tool calls, executes the matching functions,
        and returns a list of results.
        """
        results = [
            {
                "role": "assistant",
                "tool_calls": tool_calls,
            }
        ]

        if not tool_calls:
            return results

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            fn_args_str = tool_call.function.arguments

            if fn_name not in self._functions:
                print(f"Error: Function {fn_name} not found in toolbox.")
                continue

            try:
                # Parse JSON arguments from LLM
                fn_args = json.loads(fn_args_str)
                logger.info(
                    f"Invoking tool: {fn_name}({', '.join(f'{k}={v}' for k, v in fn_args.items())})"
                )

                # Execute the actual Python function
                function_to_call = self._functions[fn_name]
                output = function_to_call(**fn_args)

                # Store result (useful if you need to send it back to the LLM)
                results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": fn_name,
                        "content": str(output),
                    }
                )

            except Exception as e:
                # Store result (useful if you need to send it back to the LLM)
                results.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": fn_name,
                        "content": f"Error executing {fn_name}: {e}",
                    }
                )

        return results
