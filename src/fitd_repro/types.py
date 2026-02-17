from dataclasses import dataclass
from typing import Literal, TypedDict


Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


@dataclass(slots=True)
class PromptExample:
    example_id: str
    goal: str
    target: str | None = None
