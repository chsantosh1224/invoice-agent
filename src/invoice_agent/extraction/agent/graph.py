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
        submission: Outcome of the latest submit_invoice call, written by check_submit
            ('valid' / 'invalid') and reset to 'none' by decide at the start of every
            turn, so the router never acts on a stale outcome.
        submitted: The draft passed to submit_invoice, once valid.
        notes: The agent's audit notes from submit_invoice.
        result: The final ExtractionResult.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    doc_id: str
    run_id: str
    steps: int
    submission: Literal["none", "valid", "invalid"]
    submitted: RawInvoice
    notes: list[str]
    result: ExtractionResult


class ExtractionAgent:
    """The extraction agent: configuration in __init__, one method per graph node.

    LangGraph calls every node with only the state, so anything else a node needs (the
    model, step limit, gate threshold) is stored on `self` and reached through the
    bound method (`self.decide` remembers its `self`).

    Usage:
        graph = ExtractionAgent(model, settings, llm_factory=get_chat_model).build()
        final_state = graph.invoke({"messages": [...], "doc_id": ..., "steps": 0})
    """

    def __init__(
        self,
        model: str,
        settings: Settings,
        *,
        llm_factory: Callable[[str, Settings], BaseChatModel],
    ) -> None:
        """Prepare the model clients and settings the nodes will use.

        Args:
            model: OpenRouter slug of the agent model (must support tool calling).
            settings: App settings (step limit, gate threshold).
            llm_factory: Creates the chat model; tests pass a scripted fake.
        """
        self.model = model
        self.settings = settings
        self.max_steps = settings.agent_max_steps
        llm = llm_factory(model, settings)
        # Normal turns: the model may call any tool, one per turn.
        self.llm_free = llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)
        # Last turn: the model must call submit_invoice.
        self.llm_forced = llm.bind_tools(
            ALL_TOOLS, parallel_tool_calls=False, tool_choice="submit_invoice"
        )

    # ---------- nodes: each receives the state and returns the keys it changes ----------

    def decide(self, state: AgentState) -> dict[str, Any]:
        """One model turn: pick the next tool (forced to submit on the last turn).

        Args:
            state: Current agent state; the whole conversation is sent to the model.

        Returns:
            The model's reply (appended to messages), the updated step count, and
            `submission` reset to 'none' (a new turn has no submission outcome yet).
        """
        steps = state.get("steps", 0) + 1
        last = steps >= self.max_steps
        messages: list[AnyMessage] = [SystemMessage(AGENT_SYSTEM_PROMPT), *state["messages"]]
        if last:
            messages.append(HumanMessage(LAST_STEP_NUDGE))  # ephemeral: not stored in state
        try:
            reply = (self.llm_forced if last else self.llm_free).invoke(messages)
        except Exception as e:  # some providers reject forced tool_choice; retry unforced
            if not last:
                raise
            log.warning("agent.forced_choice_failed", error=str(e)[:120])
            reply = self.llm_free.invoke(messages)
        assert isinstance(reply, AIMessage)
        log.info(
            "agent.step",
            run_id=state.get("run_id"),
            step=steps,
            model=self.model,
            tools=[c["name"] for c in reply.tool_calls],
        )
        update: dict[str, Any] = {"messages": [reply], "steps": steps, "submission": "none"}
        if not reply.tool_calls:
            update["messages"].append(HumanMessage(NO_TOOL_NUDGE))
        return update

    def check_submit(self, state: AgentState) -> dict[str, Any]:
        """Parse the submitted draft; an invalid one goes back to the model with the error.

        Args:
            state: Agent state whose last message is a submit_invoice call.

        Returns:
            `submission`='valid' with the draft and notes; or `submission`='invalid' with
            a tool message telling the model what was wrong.
        """
        call = state["messages"][-1].tool_calls[0]  # type: ignore[union-attr]
        try:
            draft = RawInvoice.model_validate(call["args"].get("draft"))
        except ValidationError as e:
            msg = f"Submission rejected - the draft does not match the schema: {e.errors()[:5]}"
            return {
                "submission": "invalid",
                "messages": [ToolMessage(msg, tool_call_id=call["id"])],
            }
        raw_notes = call["args"].get("notes") or []
        # Some models send one string instead of a list; iterating it would split it into
        # single characters (seen with Nemotron Super).
        notes = [raw_notes] if isinstance(raw_notes, str) else [str(n) for n in raw_notes]
        return {
            "submission": "valid",
            "submitted": draft,
            "notes": notes,
            "messages": [ToolMessage("submitted", tool_call_id=call["id"])],
        }

    def finalize(self, state: AgentState) -> dict[str, Any]:
        """Re-run the validation gate on the submission and package the result.

        Args:
            state: Final agent state (with or without a submission).

        Returns:
            {'result': ExtractionResult}, accepted only if the gate passes.
        """
        draft = state.get("submitted")
        common = {
            "route": "agent",
            "model": self.model,
            "prompt_version": AGENT_PROMPT_VERSION,
            "run_id": state.get("run_id", ""),
            "notes": state.get("notes", []),
        }
        if draft is None:
            issue = ValidationIssue(
                field="*",
                code="agent.no_submission",
                message=f"no valid submission within {self.max_steps} steps",
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
                threshold=self.settings.gate_threshold,
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

    # ---------- routing: decide which node runs next (changes nothing) ----------

    def route(self, state: AgentState) -> Literal["tools", "check_submit", "decide", "finalize"]:
        """The agent's single router, used after both `decide` and `check_submit`.

        It reads explicit facts from the state, checked in this order:
        1. `submission` is 'valid'       -> finalize (a good draft is in)
        2. the model just asked to submit -> check_submit
        3. out of turns                   -> finalize (even if the model ignored the
                                             forced submit, or its last draft was invalid)
        4. the model asked for a tool     -> tools
        5. anything else                  -> decide (no tool call, or a rejected draft
                                             with turns left: let the model try again)

        Args:
            state: Agent state after `decide` or `check_submit`.

        Returns:
            The name of the next node.
        """
        if state.get("submission") == "valid":
            return "finalize"
        last = state["messages"][-1]
        tool_calls = last.tool_calls if isinstance(last, AIMessage) else []
        if tool_calls and tool_calls[0]["name"] == "submit_invoice":
            return "check_submit"
        if state.get("steps", 0) >= self.max_steps:
            return "finalize"
        if tool_calls:
            return "tools"
        return "decide"

    # ---------- wiring ----------

    def build(self) -> Any:
        """Wire the nodes into a graph and compile it.

        Returns:
            A compiled LangGraph runnable. The loop is the edge tools -> decide.
        """
        graph = StateGraph(AgentState)
        graph.add_node("decide", self.decide)
        graph.add_node("tools", ToolNode(READ_ONLY_TOOLS))
        graph.add_node("check_submit", self.check_submit)
        graph.add_node("finalize", self.finalize)

        graph.add_edge(START, "decide")
        # One router decides after both decision points.
        graph.add_conditional_edges(
            "decide", self.route, ["tools", "check_submit", "decide", "finalize"]
        )
        graph.add_edge("tools", "decide")  # the loop: every tool result goes back to the LLM
        graph.add_conditional_edges("check_submit", self.route, ["finalize", "decide"])
        graph.add_edge("finalize", END)
        return graph.compile()


def build_extraction_agent(
    model: str,
    settings: Settings,
    *,
    llm_factory: Callable[[str, Settings], BaseChatModel],
) -> Any:
    """Compile the extraction agent for one model (shortcut for ExtractionAgent(...).build()).

    Args:
        model: OpenRouter slug of the agent model (must support tool calling).
        settings: App settings (step limit, gate threshold).
        llm_factory: Creates the chat model; tests pass a scripted fake.

    Returns:
        A compiled graph. Invoke with {'messages': [...], 'doc_id': ..., 'run_id': ...,
        'steps': 0}; the final state's 'result' holds the ExtractionResult.
    """
    return ExtractionAgent(model, settings, llm_factory=llm_factory).build()
