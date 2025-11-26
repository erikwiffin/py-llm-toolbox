# Toolbox

A Python library for managing LLM tool execution. Easily register Python functions as tools that can be called by language models through the OpenAI API (or compatible APIs).

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

## Usage

### Basic Example

Here's a simple "Hello World" example:

```python
from toolbox import Toolbox
from openai import OpenAI

# 1. Setup Toolbox
toolbox = Toolbox()

# 2. Register Functions
@toolbox.function(description="A hello world function that greets someone")
@toolbox.parameter(name="who", type="string", description="The name of the person to greet")
@toolbox.parameter(name="loud", type="boolean", description="If true, shout the greeting", required=False)
def hello_world(who, loud=False):
    greeting = f"Hello {who}"
    if loud:
        greeting = greeting.upper() + "!!!"
    print(greeting)

# 3. Use with OpenAI API
client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="mistral-small-3.1-24b-instruct-2503",
    messages=[
        {"role": "system", "content": "Say hello. My name is Alice"},
    ],
    tools=toolbox.tools
)

# 4. Execute tool calls
toolbox.execute(response.choices[0].message)
```

### Key Concepts

1. **Register Functions**: Use `@toolbox.function()` to register a Python function as a tool
2. **Define Parameters**: Use `@toolbox.parameter()` to define function parameters with types and descriptions
3. **Get Tools Schema**: Access `toolbox.tools` to get the JSON schema for the OpenAI API
4. **Execute Tool Calls**: Use `toolbox.execute()` to execute tool calls from LLM responses

## Testing

Run tests with:

```bash
pytest tests/
```