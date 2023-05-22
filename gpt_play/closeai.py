from enum import Enum
from typing import Optional

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
