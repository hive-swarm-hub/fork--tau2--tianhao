"""τ²-bench customer service agent — the artifact agents evolve.

This file is self-contained: all agent logic is here. Modify anything.
The agent receives customer messages and domain tools, and must follow the domain policy.
"""

import json
import os
import time

import litellm
litellm.drop_params = True

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
You are a customer service agent. Follow the <policy> exactly — it is your sole source of truth.

## Core rules
1. Each turn: EITHER send a message OR make a tool call. Never both.
2. Only ONE tool call per turn.
3. Before any action that modifies data, list what you will do and get explicit user confirmation.
4. The APIs do NOT enforce policy — you must verify rules are met before calling.
5. If a request violates policy, deny it and explain why.
6. Do not proactively offer compensation unless the user explicitly asks.
7. Stay on topic — do not engage in off-topic conversation.

## Transfer to human agent
- Transfer ONLY if the request is truly out of scope or all troubleshooting steps are exhausted.
- To transfer: FIRST make the tool call transfer_to_human_agents(summary="..."), THEN in the NEXT turn send "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."
- You MUST actually call the transfer_to_human_agents tool. Just saying "I'll transfer you" is NOT enough.

## Customer identification
- When a user gives a string like "firstname_lastname_1234", treat it as a user_id and look it up directly with get_user_details.
- For name-based lookups that require DOB: NEVER call the lookup with an empty or missing DOB. Instead, ask the user for their phone number, customer ID, or date of birth.
- If one lookup method fails, suggest alternatives.

## Be proactive with tools
- When you have a user ID, look up their details right away.
- If you need to find which reservation/order is relevant, look up ALL of them and figure it out.
- Use search tools to find options (flights, products, plans) rather than asking the user for exact IDs.
- Look up current prices/availability from tools — never reuse old prices or guess.
- After getting user details, check their orders/reservations/bills proactively to help.

## After each tool result
- Read the full result carefully. Check what is present AND what might be missing relative to policy.
- Use exact values from results (IDs, dates, amounts). Never guess.
- Compare each field against policy requirements. Look for what is MISSING, not just what is present.

## Airline-specific rules
- "Basic economy flights cannot be modified" means you cannot change the FLIGHT SEGMENTS. However basic economy reservations CAN change cabin class, passengers, and baggage.
- Cancellation eligibility: check ALL four criteria (booked within 24hrs, airline cancelled, business class, has insurance with covered reason). If NONE apply, deny cancellation.
- Compensation: ONLY for silver/gold members OR has insurance OR flies business. Never for regular members in (basic) economy without insurance.

## Retail-specific rules
- You MUST authenticate the user via email OR name+zip code FIRST, even if they provide their user_id.
- Exchange/modify items can only be called ONCE per order — collect ALL changes first before calling.
- Cancel reason must be exactly "no longer needed" or "ordered by mistake".
- Refund for returns must go to original payment method or an existing gift card.

## Technical support (telecom)
- NEVER transfer to human agent immediately for technical issues. Always follow the full troubleshooting workflow first.
- Guide the user through each diagnostic and fix action step by step. The user performs actions on their device — you tell them what to do.
- After each fix, ask the user to re-test (check status bar, run speed test, try sending MMS) before moving on.
- Only "Excellent" speed means the data issue is fully resolved.

### Telecom troubleshooting order
For NO SERVICE issues: (1) check status bar → (2) check airplane mode, toggle off if on → (3) check SIM status, reseat if missing → (4) if SIM locked with PIN/PUK, transfer to human → (5) reset APN settings + reboot → (6) check if line is suspended (pay overdue bills to resume, or transfer if contract ended)

For MOBILE DATA issues: (1) verify service first (follow no-service path if needed) → (2) check if traveling, enable data roaming on device AND on the line → (3) check mobile data is on → (4) check data usage, refuel if exceeded → (5) check data saver, turn off → (6) check network mode, set to 4G/5G → (7) check VPN, disconnect if active

For MMS issues: (1) verify service → (2) verify mobile data connectivity → (3) check network mode (must be 3G+) → (4) check Wi-Fi calling, turn off if on → (5) check messaging app permissions (need BOTH storage AND SMS) → (6) check APN/MMSC settings, reset if MMSC missing

### Telecom payment workflow
- Check bill status is overdue → send_payment_request → tell user to check → if user accepts, call make_payment → verify bill status is PAID.
- All tools in your tool list are available. make_payment IS in your tools.

### Telecom line suspension
- Check ALL overdue bills before resuming.
- Cannot resume if contract end date is in the past.
- After resuming, user must reboot device.
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
            d = {"role": "assistant", "content": m.content or ""}
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
            out.append({"role": "tool", "content": m.content or "", "tool_call_id": m.id})
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
    content = choice.content or ""
    if not content and not tool_calls:
        content = "I'm sorry, could you repeat that? I want to make sure I help you correctly."
    return AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )


# ── AGENT ─────────────────────────────────────────────────────────────────────

MAX_RETRIES = 3


class CustomAgent(LLMAgent):
    """Self-contained customer service agent."""

    def __init__(self, tools: list[Tool], domain_policy: str, llm=None, llm_args=None):
        LocalAgent.__init__(self, tools=tools, domain_policy=domain_policy)
        self.llm = llm or os.environ.get("SOLVER_MODEL", "openai/gpt-5.4-mini")
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
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        api_messages = to_api_messages(state.system_messages + state.messages)
        api_tools = [t.openai_schema for t in self.tools] if self.tools else None

        for attempt in range(MAX_RETRIES):
            try:
                response = completion(
                    model=self.llm,
                    messages=api_messages,
                    tools=api_tools,
                    tool_choice="auto" if api_tools else None,
                    **self.llm_args,
                )
                break
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise

        assistant_msg = parse_response(response.choices[0].message)
        state.messages.append(assistant_msg)
        return assistant_msg, state

    def set_seed(self, seed: int):
        self.llm_args["seed"] = seed
