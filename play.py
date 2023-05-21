from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, ClassVar, Optional, Protocol

import click
import openai
from pydantic import BaseModel


class CompletionChoice(BaseModel):
    finish_reason: str
    index: int
    logprobs: Optional[int]
    text: str


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class CompletionsResponse(BaseModel):
    choices: list[CompletionChoice]
    created: int
    id: str
    model: str
    object: str
    usage: Usage

    def take_first(self) -> str:
        return self.choices[0].text


class CompletionsRequest(BaseModel):
    model: str
    prompt: Optional[str | list[str | list[str]]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None
    n: Optional[int] = 1
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[str | list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1

    def query(self) -> CompletionsResponse:
        return CompletionsResponse(**openai.Completion.create(stream=False, **self.dict()))  # type: ignore


class Role(str, Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"

    def __repr__(self) -> str:
        return str.__repr__(self.value)


class Message(BaseModel):
    role: Role
    content: str
    # name: Optional[str] = None


class ResponseChoice(BaseModel):
    finish_reason: str
    index: int
    logprobs: Optional[int]
    message: Message


class ChatResponse(BaseModel):
    choices: list[ResponseChoice]
    created: int
    id: str
    model: str
    object: str
    usage: Usage

    def take_first(self) -> str:
        return self.choices[0].message.content


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[str | list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0

    def query(self) -> ChatResponse:
        return ChatResponse(**openai.ChatCompletion.create(stream=False, **self.dict()))  # type: ignore


class EditChoice(BaseModel):
    index: int
    text: str


class EditResponse(BaseModel):
    choices: list[EditChoice]
    created: int
    object: str
    usage: Usage

    def take_first(self) -> str:
        return self.choices[0].text


class EditRequest(BaseModel):
    model: str
    input: Optional[str] = None
    instruction: str
    n: Optional[int] = 1
    temperature: Optional[float] = 0.0
    top_p: Optional[float] = None

    def query(self) -> EditResponse:
        return EditResponse(**openai.Edit.create(**self.dict()))  # type: ignore


def validate_tool(answer: str, tools: list[str]) -> Optional[tuple[str, str]]:
    """Validate that the answer is a valid tool response as follows:
    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [list of tools]
    Input: the input to the action
    ```
    """
    answer = answer.strip()
    lines = answer.split("\n")
    if len(lines) != 3:
        return
    if lines[0] != "Thought: Do I need to use a tool? Yes":
        return
    if lines[1].split(":")[0] != "Action":
        return
    if lines[2].split(":")[0] != "Input":
        return
    if lines[1].split(":")[1].strip() not in tools:
        return
    return lines[1].split(":")[1].strip(), lines[2].split(":")[1].strip()


def validate_ai(answer) -> Optional[str]:
    """Validate that the answer is a valid tool response as follows:

    ```
    Thought: Do I need to use a tool? No
    AI: your response here
    ```
    """
    answer = answer.strip()
    lines = answer.split("\n")
    if len(lines) != 2:
        return
    if lines[0] != "Thought: Do I need to use a tool? No":
        return
    a, b = lines[1].split(":")
    if a != "AI":
        return
    return b.strip()


class Tool(BaseModel):
    name: str
    description: str
    run: Callable[[str], str]

    # @abstractmethod
    # def run(self, query: str) -> str:
    #     raise NotImplementedError


class ToolStore(BaseModel):
    tools: ClassVar[list[Tool]] = []

    @classmethod
    def get_tool(cls, name: str) -> Optional[Tool]:
        for tool in cls.tools:
            if tool.name == name:
                return tool
        return None


def regester_tool(func: Callable[[str], str]) -> None:
    tool = Tool(
        name=func.__name__.capitalize(),
        description=func.__doc__ or func.__name__,
        run=func,
    )
    ToolStore.tools.append(tool)


@regester_tool
def calculator(query: str) -> str:
    """A calculator. Useful for when you need to answer questions about math."""
    return str(eval(query))


@click.command()
@click.option("--query", "-q", default="Hello, world!")
@click.option("--model", "-m", default="text-davinci-003")
def main(query: str, model: str) -> None:
    prompt = """
Have a conversation with a human, answering the following questions as best you can. You have access to the following tool
s:
    
> Search: A search engine. Useful for when you need to answer questions about current events. Input should be a search q
uery.
> Calculator: Useful for when you need to answer questions about math.

You can chose either to use a tool, or to respond to the human directly.

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [Search, Calculator]
Input: the input to the action
Observation: the observation of the action
```

If you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
AI: your response here
```

Begin!

{query}
"""
    # {chat_history}
    # {agent_scratchpad}
    raw_answer = (
        CompletionsRequest(
            model=model,
            prompt=prompt.format(query=query),
            stop=["Observation:"],
        )
        .query()
        .take_first()
    )
    # print(raw_answer)
    if out := validate_tool(raw_answer, ["Search", "Calculator"]):
        tool, input = out
        match ToolStore.get_tool(tool):
            case None:
                print(f"Unknown tool: {tool}")
            case tool:
                print(tool.run(input))
    elif out := validate_ai(raw_answer):
        print(out)


if __name__ == "__main__":
    main()
