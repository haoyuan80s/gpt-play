from typing import Optional

import click

from gpt_play.closeai import CompletionsRequest
from gpt_play.tool_store import ToolStore
from gpt_play.tools import calculator


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


@click.command()
@click.option("--query", "-q", default="Hello, world!")
@click.option("--model", "-m", default="text-davinci-003")
@click.option("--verbose", "-v", is_flag=True)
def main(query: str, model: str, verbose: bool) -> None:
    tool_store = ToolStore()
    tool_store.regester_tool(calculator)
    prompt = """
Have a conversation with a human, answering the following questions as best you can.
You have access to the following tools: 
{tool_prompt}
You can chose either to use a tool, or to respond to the human directly.
- when using a tool, use the following format:
```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of {tool_names}
Input: <the input to the action>
Observation: <the observation of the action>
```
- when not using a tool:
```
Thought: Do I need to use a tool? No
AI: <your response here>
```
Begin!
---
{query}
"""
    # {chat_history}
    # {agent_scratchpad}
    raw_answer = (
        CompletionsRequest(
            model=model,
            prompt=prompt.format(
                tool_prompt=tool_store.tool_prompt,
                tool_names=tool_store.tool_names,
                query=query,
            ),
            stop=["Observation:"],
        )
        .query()
        .take_first()
    )
    if verbose:
        print(raw_answer)
    if out := validate_tool(raw_answer, ["Search", "Calculator"]):
        tool, input = out
        match tool_store.get_tool(tool):
            case None:
                print(f"Unknown tool: {tool}")
            case tool:
                print(tool.run(input))
    elif out := validate_ai(raw_answer):
        print(out)


if __name__ == "__main__":
    main()
