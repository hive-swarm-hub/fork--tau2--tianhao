"""τ²-bench customer service agent — the artifact agents evolve.

This file is self-contained: all agent logic is here. Modify anything.
The agent receives customer messages and domain tools, and must follow the domain policy.
"""

import json
import os

from litellm import completion

from tau2.agent.base import LocalAgent, ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool

# ── PROMPT (the main lever for improving performance) ─────────────────────────

INSTRUCTIONS = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

SYSTEM_TEMPLATE = """
<instructions>
{instructions}
</instructions>
<policy>
{policy}
</policy>
""".strip()

# ── MESSAGE CONVERSION ────────────────────────────────────────────────────────

def to_api_messages(messages):
    """Convert tau2 message objects to OpenAI-style dicts."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            d = {"role": "assistant", "content": m.content}
            if m.is_tool_call():
                d["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in m.tool_calls
                ]
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content, "tool_call_id": m.id})
    return out


def parse_response(choice):
    """Convert an LLM API response choice into a tau2 AssistantMessage."""
    tool_calls = None
    if choice.tool_calls:
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in choice.tool_calls
        ]
    return AssistantMessage(
        role="assistant",
        content=choice.content,
        tool_calls=tool_calls or None,
    )


# ── AGENT ─────────────────────────────────────────────────────────────────────

class CustomAgent(LLMAgent):
    """Self-contained customer service agent.

    Extends LLMAgent for compatibility with tau2's run_task() constructor,
    but all logic is overridden here — nothing is hidden.
    """

    def __init__(self, tools: list[Tool], domain_policy: str, llm=None, llm_args=None):
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.llm = llm or os.environ.get("SOLVER_MODEL", "gpt-5.4-mini")
        self.llm_args = dict(llm_args or {})

    @property
    def system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(instructions=INSTRUCTIONS, policy=self.domain_policy)

    def get_init_state(self, message_history=None) -> LLMAgentState:
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=list(message_history or []),
        )

    def generate_next_message(self, message: ValidAgentInputMessage, state: LLMAgentState):
        # 1. Append incoming message(s) to conversation history
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # 2. Build API request
        api_messages = to_api_messages(state.system_messages + state.messages)
        api_tools = [t.openai_schema for t in self.tools] if self.tools else None

        # 3. Call LLM
        response = completion(
            model=self.llm,
            messages=api_messages,
            tools=api_tools,
            tool_choice="auto" if api_tools else None,
            **self.llm_args,
        )

        # 4. Parse response
        assistant_msg = parse_response(response.choices[0].message)
        state.messages.append(assistant_msg)
        return assistant_msg, state

    def set_seed(self, seed: int):
        self.llm_args["seed"] = seed
