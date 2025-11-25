import json
import logging
from functools import wraps

logger = logging.getLogger(__name__)


class Toolbox:
    def __init__(self):
        # Maps function names to the actual python callable
        self._functions = {}
        # Stores the JSON schemas expected by OpenAI
        self._tools_schema = []

    @property
    def tools(self):
        """Returns the list of tool definitions for the OpenAI API."""
        return self._tools_schema

    def parameter(self, name, type, description, required=True, enum=None):
        """
        Decorator to define a parameter for the tool.
        Since decorators run bottom-to-top, these attach metadata to the function
        before @toolbox.function registers it.
        """

        def decorator(func):
            if not hasattr(func, "_tool_params"):
                func._tool_params = {}

            func._tool_params[name] = {"type": type, "description": description}

            if enum:
                func._tool_params[name]["enum"] = enum

            if not hasattr(func, "_tool_required"):
                func._tool_required = []

            if required:
                func._tool_required.append(name)

            return func

        return decorator

    def function(self, description):
        """
        Decorator to register the function as a tool.
        This must be placed *above* @toolbox.parameter decorators.
        """

        def decorator(func):
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
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def execute(self, tool_calls):
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

            logger.info(f" -> Invoking tool: {fn_name}")

            if fn_name not in self._functions:
                print(f"Error: Function {fn_name} not found in toolbox.")
                continue

            try:
                # Parse JSON arguments from LLM
                fn_args = json.loads(fn_args_str)

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
