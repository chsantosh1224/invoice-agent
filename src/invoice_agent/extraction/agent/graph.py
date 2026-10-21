"""The extraction agent as a LangGraph state graph.

    START -> decide --(tool call)--> tools --> decide ...
                    --(submit)-----> check_submit --(valid)--> finalize -> END
                                                  --(invalid)-> decide
                    --(step limit / no submission)-----------> finalize

`decide` is the LLM choosing the next action; `tools` executes read-only tools;
`finalize` re-runs the same validation gate as the fast path, so the agent never
certifies its own work.
"""

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import ValidationError

from invoice_agent.config import Settings
from invoice_agent.extraction.agent import workspace
from invoice_agent.extraction.agent.prompts import (
    AGENT_PROMPT_VERSION,
    AGENT_SYSTEM_PROMPT,
    LAST_STEP_NUDGE,
    NO_TOOL_NUDGE,
)
from invoice_agent.extraction.agent.tools import ALL_TOOLS, READ_ONLY_TOOLS
from invoice_agent.extraction.validate import validate
from invoice_agent.normalize import to_invoice
from invoice_agent.schemas import ExtractionResult, Invoice, RawInvoice, ValidationIssue

log = structlog.get_logger()


class AgentState(TypedDict, total=False):
    """The agent's state. Deliberately small and serializable (no documents or images).

    Attributes:
        messages: The conversation (task, model turns, tool results).
        doc_id: Workspace id; tools use it to reach the document.
        run_id: Correlates log lines with the fast-path run.
        steps: Model turns taken so far.
        submitted: The draft passed to submit_invoice, once valid.
        notes: The agent's audit notes from submit_invoice.
        result: The final ExtractionResult.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    doc_id: str
    run_id: str
    steps: int
    submitted: RawInvoice
    notes: list[str]
    result: ExtractionResult


def build_extraction_agent(
    model: str,
    settings: Settings,
    *,
    llm_factory: Callable[[str, Settings], BaseChatModel],
) -> Any:
    """Compile the extraction agent for one model.

    Args:
        model: OpenRouter slug of the agent model (must support tool calling).
        settings: App settings (step limit, gate threshold).
        llm_factory: Creates the chat model; tests pass a scripted fake.

    Returns:
        A compiled graph. Invoke with {'messages': [...], 'doc_id': ..., 'run_id': ...,
        'steps': 0}; the final state's 'result' holds the ExtractionResult.
    """
    max_steps = settings.agent_max_steps
    llm = llm_factory(model, settings)
    free = llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)
    forced = llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False, tool_choice="submit_invoice")

    def decide(state: AgentState) -> dict[str, Any]:
        """One model turn: pick the next tool (forced to submit on the last turn)."""
        steps = state.get("steps", 0) + 1
        last = steps >= max_steps
        messages: list[AnyMessage] = [SystemMessage(AGENT_SYSTEM_PROMPT), *state["messages"]]
        if last:
            messages.append(HumanMessage(LAST_STEP_NUDGE))  # ephemeral: not stored in state
        try:
            reply = (forced if last else free).invoke(messages)
        except Exception as e:  # some providers reject forced tool_choice; retry unforced
            if not last:
                raise
            log.warning("agent.forced_choice_failed", error=str(e)[:120])
            reply = free.invoke(messages)
        assert isinstance(reply, AIMessage)
        log.info(
            "agent.step",
            run_id=state.get("run_id"),
            step=steps,
            model=model,
            tools=[c["name"] for c in reply.tool_calls],
        )
        update: dict[str, Any] = {"messages": [reply], "steps": steps}
        if not reply.tool_calls:
            update["messages"].append(HumanMessage(NO_TOOL_NUDGE))
        return update

    def route(state: AgentState) -> Literal["tools", "check_submit", "decide", "finalize"]:
        """Send tool calls to the tools, submissions to validation, and enforce the limit.

        Once the step budget is spent, anything but a submission ends the run - even if
        the model ignored the forced submit and asked for another tool.
        """
        last = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
        out_of_steps = state.get("steps", 0) >= max_steps
        if last.tool_calls and last.tool_calls[0]["name"] == "submit_invoice":
            return "check_submit"
        if out_of_steps:
            return "finalize"
        return "tools" if last.tool_calls else "decide"

    def check_submit(state: AgentState) -> dict[str, Any]:
        """Parse the submitted draft; an invalid one goes back to the model with the error."""
        call = state["messages"][-1].tool_calls[0]  # type: ignore[union-attr]
        try:
            draft = RawInvoice.model_validate(call["args"].get("draft"))
        except ValidationError as e:
            msg = f"Submission rejected - the draft does not match the schema: {e.errors()[:5]}"
            return {"messages": [ToolMessage(msg, tool_call_id=call["id"])]}
        raw_notes = call["args"].get("notes") or []
        # Some models send one string instead of a list; iterating it would split it into
        # single characters (seen with Nemotron Super).
        notes = [raw_notes] if isinstance(raw_notes, str) else [str(n) for n in raw_notes]
        return {
            "submitted": draft,
            "notes": notes,
            "messages": [ToolMessage("submitted", tool_call_id=call["id"])],
        }

    def after_submit(state: AgentState) -> Literal["finalize", "decide"]:
        """Finish once a valid draft is in; otherwise let the model try again if it can."""
        if state.get("submitted") is not None or state.get("steps", 0) >= max_steps:
            return "finalize"
        return "decide"

    def finalize(state: AgentState) -> dict[str, Any]:
        """Re-run the validation gate on the submission and package the result."""
        draft = state.get("submitted")
        common = {
            "route": "agent",
            "model": model,
            "prompt_version": AGENT_PROMPT_VERSION,
            "run_id": state.get("run_id", ""),
            "notes": state.get("notes", []),
        }
        if draft is None:
            issue = ValidationIssue(
                field="*",
                code="agent.no_submission",
                message=f"no valid submission within {max_steps} steps",
            )
            result = ExtractionResult(
                status="needs_human_review", invoice=Invoice(), issues=[issue], **common
            )
        else:
            invoice, issues = to_invoice(draft)
            gate = validate(
                invoice,
                workspace.get(state["doc_id"]).doc,
                normalize_issues=issues,
                threshold=settings.gate_threshold,
            )
            result = ExtractionResult(
                status="accepted" if gate.passed else "needs_human_review",
                invoice=invoice,
                raw=draft,
                issues=gate.issues,
                field_confidence=gate.field_confidence,
                score=gate.score,
                **common,
            )
        log.info(
            "agent.done",
            run_id=state.get("run_id"),
            status=result.status,
            steps=state.get("steps"),
            notes=result.notes,
        )
        return {"result": result}

    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("tools", ToolNode(READ_ONLY_TOOLS))
    graph.add_node("check_submit", check_submit)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route, ["tools", "check_submit", "decide", "finalize"])
    graph.add_edge("tools", "decide")
    graph.add_conditional_edges("check_submit", after_submit, ["finalize", "decide"])
    graph.add_edge("finalize", END)
    return graph.compile()
