from collections.abc import Iterable
from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Callable


@dataclass
class Parameter:
    """Represents a parameter for a tool function."""

    name: str
    type: str
    description: str | None = None
    required: bool = True
    enum: list[Any] | None = None  # pyright: ignore[reportExplicitAny]


@dataclass
class Function:
    """Represents a tool function with its metadata."""

    name: str = ""
    description: str = ""
    callable: Callable[..., Any] | None = None  # pyright: ignore[reportExplicitAny]
    parameters: list[Parameter] = field(default_factory=list)


def python_type_to_json_schema_type(python_type: type) -> str:
    """
    Converts Python type annotations to JSON schema type strings.
    """
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


def build_tools_schema(functions: Iterable[Function]) -> list[dict[str, Any]]:
    """
    Builds the OpenAI tool schema from a list of Function objects.

    Args:
        functions: List of Function objects to convert to tool definitions

    Returns:
        List of tool definitions in OpenAI format
    """
    schema: list[dict[str, Any]] = []
    for func_data in functions:
        # Only include functions that have been fully registered (have name and description)
        if not func_data.name or not func_data.description:
            continue

        # Build properties dict from Parameter objects
        properties = {}
        required: list[str] = []

        for param in func_data.parameters:
            param_dict: dict[str, Any] = {"type": param.type}
            if param.description:
                param_dict["description"] = param.description
            if param.enum:
                param_dict["enum"] = param.enum
            properties[param.name] = param_dict

            if param.required:
                required.append(param.name)

        tool_definition: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": func_data.name,
                "description": func_data.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        schema.append(tool_definition)

    return schema
