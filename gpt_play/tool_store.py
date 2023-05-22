from typing import Callable, Optional

from pydantic import BaseModel


class Tool(BaseModel):
    name: str
    description: str
    run: Callable[[str], str]


class ToolStore(BaseModel):
    tools: list[Tool] = []

    def get_tool(self, name: str) -> Optional[Tool]:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def add_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    def regester_tool(self, func: Callable[[str], str]) -> None:
        tool = Tool(
            name=func.__name__.capitalize(),
            description=func.__doc__ or func.__name__,
            run=func,
        )
        self.add_tool(tool)

    @property
    def tool_prompt(self) -> str:
        return "\n".join([f"> {tool.name}: {tool.description}" for tool in self.tools])

    @property
    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]
