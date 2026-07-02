import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Type
import click
from stratasift.utils.ui import stream_supervisor_thought, log_event, AnalysisSpinner
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from stratasift.config import get_runtime_config
from stratasift.graph.state import PaperIngestionState
from stratasift.core.models import AtomicInsight
from stratasift.utils.file_io import format_insight, generate_insight_id
from stratasift.utils.embeddings import SimpleEmbeddingFunction, cosine_similarity
import datetime


# Define Pydantic Schemas for Structured Outputs


class TriageDecision(BaseModel):
    domain_relevance: float = Field(
        default=0.0,
        description="Score between 0.0 and 1.0 indicating domain subject-matter alignment.",
    )
    methodology_relevance: float = Field(
        default=0.0,
        description="Score between 0.0 and 1.0 indicating technical methodology alignment.",
    )
    central_hypothesis: str = Field(
        default="",
        description="The paper's core scientific contribution as a hypothesis statement.",
    )
    reading_directive: str = Field(
        default="",
        description="Reading directive for the specialist node if relevant, otherwise empty.",
    )


class SpecialistExtraction(BaseModel):
    specialist_type: str = Field(
        default="methodology",
        description="Either 'methodology' or 'data'.",
    )
    data_points: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted key details, parameters, yields, or protocols.",
    )
    source_quotes: List[str] = Field(
        default_factory=list,
        description="Exact verbatim quotes from the source text supporting these data points.",
    )


class ReflectionDecision(BaseModel):
    verified: bool = Field(
        default=True,
        description="True if no discrepancies or hallucinations are found, False otherwise.",
    )
    feedback: str = Field(
        default="",
        description="Feedback message detailing the discrepancy if verified is False, otherwise empty.",
    )
    insight_dossier: str = Field(
        default="",
        description="A brief cross-reference summary of verified facts if verified is True.",
    )


class NetworkDecision(BaseModel):
    decision: str = Field(description="Must be 'append', 'contradict', or 'create'.")
    target_note: str = Field(
        description="Title of the target note in vault if decision is 'append' or 'contradict', otherwise empty."
    )
    note_title: str = Field(description="Title of the new or updated note.")
    markdown_content: str = Field(
        description="The complete synthesised Obsidian note in markdown format with [[WikiLinks]]."
    )


class AtomicInsightList(BaseModel):
    insights: List[AtomicInsight] = Field(
        ..., description="List of synthesised atomic insights."
    )


# --- Mock LLM Fallback Implementation ---


class MockStructuredModel:
    """Mock structured model that returns mock Pydantic instances matching the schema."""

    def __init__(self, schema: Type[BaseModel]):
        self.schema = schema

    def invoke(self, messages, config=None, **kwargs) -> Any:
        prompt_text = ""
        for m in messages:
            if hasattr(m, "content"):
                prompt_text += "\n" + str(m.content)
            else:
                prompt_text += "\n" + str(m)

        prompt_lower = prompt_text.lower()

        if self.schema == TriageDecision:
            if "broken" in prompt_lower:
                return TriageDecision(
                    domain_relevance=0.2,
                    methodology_relevance=0.1,
                    central_hypothesis="",
                    reading_directive="",
                )
            elif "battery" in prompt_lower:
                return TriageDecision(
                    domain_relevance=0.9,
                    methodology_relevance=0.85,
                    central_hypothesis="Solid-state LLZO electrolyte sintering at 800°C achieves optimal ionic conductivity for high-energy density storage.",
                    reading_directive="Extract solid-state battery synthesis methodologies, focusing on calcination temperatures and yields.",
                )
            else:  # biomass or others
                return TriageDecision(
                    domain_relevance=0.4,
                    methodology_relevance=0.85,
                    central_hypothesis="Pyrolysis conditions of pine wood biomass determine biochar yield distribution.",
                    reading_directive="Extract biomass pyrolysis conditions and carbon yields.",
                )

        elif self.schema == SpecialistExtraction:
            if "methodology" in prompt_lower:
                return SpecialistExtraction(
                    specialist_type="methodology",
                    data_points={
                        "electrolyte": "LLZO (Li7La3Zr2O12)",
                        "sintering_protocol": "800°C for 6 hours",
                        "calcination_temperature": "800°C",
                    },
                    source_quotes=[
                        "The sintering was performed at 800°C for 6h in air",
                        "LLZO electrolyte was chosen",
                    ],
                )
            else:  # data / results
                # Simulate the discrepancy retry loop
                if (
                    "retry" in prompt_lower
                    or "discrepancy" in prompt_lower
                    or "feedback" in prompt_lower
                    or "loop 1" in prompt_lower
                    or "incorrect" in prompt_lower
                ):
                    return SpecialistExtraction(
                        specialist_type="data",
                        data_points={
                            "ionic_conductivity": "1.2e-4 S/cm",
                            "yield": "85%",
                        },
                        source_quotes=[
                            "ionic conductivity of 1.2e-4 S/cm",
                            "yielding an overall 85% conversion",
                        ],
                    )
                else:
                    return SpecialistExtraction(
                        specialist_type="data",
                        data_points={
                            "ionic_conductivity": "1.2e-4 S/cm",
                            "yield": "95%",
                        },
                        source_quotes=[
                            "ionic conductivity of 1.2e-4 S/cm",
                            "yielding an overall 85% conversion",
                        ],
                    )

        elif self.schema == ReflectionDecision:
            if "95%" in prompt_lower and not (
                "retry count: 1" in prompt_lower
                or "retry_count: 1" in prompt_lower
                or "retry_count:1" in prompt_lower
            ):
                return ReflectionDecision(
                    verified=False,
                    feedback="Discrepancy found in Yield metric: extracted 95% but results section states 85%.",
                    insight_dossier="",
                )
            else:
                if "biomass" in prompt_lower:
                    dossier = "Cross-reference verified: Pyrolysis parameters and biochar yields match source text."
                else:
                    dossier = "Cross-reference verified: LLZO sintering parameters, ionic conductivity, and yield metrics match source text."
                return ReflectionDecision(
                    verified=True, feedback="", insight_dossier=dossier
                )

        elif self.schema == AtomicInsightList:
            if "biomass" in prompt_lower:
                return AtomicInsightList(
                    insights=[
                        AtomicInsight(
                            title="Biomass pyrolysis yields biochar and bio-oil under nitrogen atmosphere",
                            core_insight="Pyrolysis of pine wood biomass at 500°C under nitrogen atmosphere. Slower heating rates increase biochar yield to 65%, while rapid heating increases bio-oil fractions. This provides an efficient framework for converting raw wood sources into usable biofuel products.",
                            context_evidence="**[Source: variant_case_biomass.md]**\n* **Data Point:** Pyrolysis at 500°C\n* **Example:** Biochar yield 65%",
                            related_vectors=["variant_case_biomass"],
                        )
                    ]
                )
            else:
                if (
                    "retry" in prompt_lower
                    or "discrepancy" in prompt_lower
                    or "feedback" in prompt_lower
                    or "incorrect" in prompt_lower
                ):
                    # After retry correction — yield corrected to 85%
                    return AtomicInsightList(
                        insights=[
                            AtomicInsight(
                                title="Sintering LLZO solid-state electrolyte yields high ionic conductivity",
                                core_insight="Solid-state battery synthesis using LLZO electrolyte. Sintering protocol at 800°C for 6 hours yields a cubic phase with ionic conductivity of 1.2e-4 S/cm and an overall yield of 85%. This achieves optimal performance parameters required for high-energy density storage applications.",
                                context_evidence="**[Source: success_case_battery.md]**\n* **Aggregated Insight from success_case_battery.md:** Sintering protocol at 800°C for 6 hours yields a cubic phase with ionic conductivity of 1.2e-4 S/cm and an overall yield of 85%. [[success_case_battery]]",
                                related_vectors=["success_case_battery"],
                            )
                        ]
                    )
                else:
                    # First pass — yield has the intentional discrepancy (95%)
                    return AtomicInsightList(
                        insights=[
                            AtomicInsight(
                                title="LLZO electrolyte conductivity and yield metrics analysis for final compound",
                                core_insight="Analysis of final LLZO compound performance through systematic observation. The synthesised compound possessed an impressive ionic conductivity of 1.2e-4 S/cm while also yielding an overall 95% conversion rate. This distinct combination demonstrates strong potential for high-capacity energy storage solutions.",
                                context_evidence="Bulleted summary of data findings.",
                                related_vectors=["success_case_battery"],
                            )
                        ]
                    )

        elif self.schema == NetworkDecision:
            if "biomass" in prompt_lower:
                return NetworkDecision(
                    decision="create",
                    target_note="",
                    note_title="Biomass Pyrolysis Yields",
                    markdown_content="""# Biomass Pyrolysis Yields

## Abstract
Synthesised biomass dossier: Pyrolysis of pine wood biomass at 500°C under nitrogen atmosphere. Slower heating rates increase biochar yield to 65%, while rapid heating increases bio-oil fractions.

## Ingestion Linkages
* [[variant_case_biomass]]
""",
                )
            else:
                return NetworkDecision(
                    decision="append",
                    target_note="Solid-State Calcination Limits.md",
                    note_title="Solid-State Calcination Limits.md",
                    markdown_content="""# Solid-State Calcination Limits

## Sintering protocols
Previous work shows calcination limits around 900°C.

## Aggregated Insight from success_case_battery.md
Sintering protocol at 800°C for 6 hours yields a cubic phase with ionic conductivity of 1.2e-4 S/cm and an overall yield of 85%. [[success_case_battery]]
""",
                )

        return self.schema()


class MockLLM:
    """Mock LLM to simulate ChatGoogleGenerativeAI or ChatOllama when keys are missing."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def invoke(self, messages, config=None, **kwargs) -> AIMessage:
        return AIMessage(content="Mock response")

    def with_structured_output(self, schema, **kwargs) -> MockStructuredModel:
        return MockStructuredModel(schema)


# --- Initialise Real / Mock Clients ---

supervisor_llm = None
specialist_llm = None
_last_config = None


def _build_llm(block_cfg, config):
    if os.environ.get("STRATASIFT_MOCK") == "true":
        return MockLLM(block_cfg.model)

    provider = block_cfg.provider
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        gemini_cfg = config.providers.gemini
        return ChatGoogleGenerativeAI(
            model=block_cfg.model,
            temperature=block_cfg.temperature,
            google_api_key=gemini_cfg.api_key if gemini_cfg else None,
        )
    elif provider == "ollama_cloud":
        from langchain_ollama import ChatOllama

        oc_cfg = config.providers.ollama_cloud
        return ChatOllama(
            model=block_cfg.model,
            base_url=oc_cfg.base_url if oc_cfg else "https://ollama.com",
            temperature=block_cfg.temperature,
            client_kwargs={"headers": {"Authorization": f"Bearer {oc_cfg.api_key}"}}
            if oc_cfg and oc_cfg.api_key
            else {},
        )
    elif provider == "ollama_local":
        from langchain_ollama import ChatOllama

        ol_cfg = config.providers.ollama_local
        return ChatOllama(
            model=block_cfg.model,
            base_url=ol_cfg.base_url if ol_cfg else "http://localhost:11434",
            temperature=block_cfg.temperature,
        )
    else:
        return MockLLM(block_cfg.model)


def _init_llms():
    """Initialise the LLM clients for the supervisor and specialists."""
    global supervisor_llm, specialist_llm, _last_config
    try:
        config = get_runtime_config()
    except Exception:
        config = None

    if (
        supervisor_llm is not None
        and specialist_llm is not None
        and config is _last_config
    ):
        return
    _last_config = config

    if config:
        supervisor_llm = _build_llm(config.blocks.supervisor_block, config)
        specialist_llm = _build_llm(config.blocks.analysis_block, config)
    else:
        supervisor_llm = MockLLM("gemini-1.5-pro")
        specialist_llm = MockLLM("gemma4:31b-cloud")


# --- LangGraph Nodes Implementation ---


def supervisor_triage_node(state: PaperIngestionState) -> Dict[str, Any]:
    """Triage node performing dual-vector relevance scoring.

    Evaluates abstract/intro and conclusions against structured domain
    and methodology interests to determine relevance along two vectors.
    Formulates a central hypothesis and generates context-aware directives.
    """
    _init_llms()
    config = get_runtime_config()
    model = supervisor_llm.with_structured_output(TriageDecision, method="json_schema")

    source_doc = state["source_doc"]
    abstract = source_doc.abstract_intro
    conclusions = source_doc.conclusions or ""

    # Build structured profile from config
    domain_interests = config.system.domain_interests
    methodology_interests = config.system.methodology_interests

    system_instruction = config.prompts.triage
    system_instruction += (
        "\n\nYou MUST return a JSON object containing the following keys:\n"
        "- domain_relevance: A float between 0.0 and 1.0 indicating domain subject-matter alignment\n"
        "- methodology_relevance: A float between 0.0 and 1.0 indicating technical methodology alignment\n"
        "- central_hypothesis: The paper's core scientific contribution as a hypothesis statement\n"
        "- reading_directive: A clear instruction for the specialist node detailing what to extract if max(domain_relevance, methodology_relevance) >= 0.75, otherwise empty"
    )

    # Compose the human message with both abstract and conclusions
    human_content = (
        f"Domain Interests: {', '.join(domain_interests)}\n\n"
        f"Methodology Interests: {', '.join(methodology_interests)}\n\n"
        f"Abstract/Introduction:\n{abstract}"
    )
    if conclusions:
        human_content += f"\n\nConclusions:\n{conclusions}"

    prompt = [
        SystemMessage(content=system_instruction),
        HumanMessage(content=human_content),
    ]

    result = model.invoke(prompt)
    threshold = config.blocks.supervisor_block.relevance_threshold or 0.75

    # Compute effective relevance as max of the two vectors
    effective_relevance = max(result.domain_relevance, result.methodology_relevance)

    # Derive match type
    domain_passes = result.domain_relevance >= threshold
    methodology_passes = result.methodology_relevance >= threshold

    if domain_passes and methodology_passes:
        match_type = "full"
    elif methodology_passes:
        match_type = "methodology_only"
    elif domain_passes:
        match_type = "domain_only"
    else:
        match_type = ""

    # Trace log
    if effective_relevance >= threshold:
        stream_supervisor_thought(
            config.blocks.supervisor_block.model,
            f"Triage passed (Domain: {result.domain_relevance:.2f}, Methodology: {result.methodology_relevance:.2f}, Match: {match_type}). Generated reading directive.",
        )
    else:
        stream_supervisor_thought(
            config.blocks.supervisor_block.model,
            f"Triage failed (Domain: {result.domain_relevance:.2f}, Methodology: {result.methodology_relevance:.2f}). Paper is irrelevant.",
        )

    # Initialise run_id and context_db_path early if missing
    run_id = state.get("run_id") or generate_insight_id()
    from pathlib import Path

    quarantine_path = Path(config.system.quarantine_path)
    context_db_path = state.get("context_db_path") or str(
        quarantine_path / f"tmp-context-{run_id}.json"
    )

    # Write early initialisation state to the temporary Context DB JSON file
    db_path = Path(context_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_data = {
        "run_id": run_id,
        "domain_relevance": result.domain_relevance,
        "methodology_relevance": result.methodology_relevance,
        "match_type": match_type,
        "central_hypothesis": result.central_hypothesis,
        "reading_directive": result.reading_directive,
        "source_doc": {
            "title": source_doc.title,
            "metadata": source_doc.metadata,
            "abstract_intro": source_doc.abstract_intro,
        },
        "status": "triaged",
        "verified": True,
        "feedback": "",
        "insight_dossier": "",
        "insights": {},
    }
    try:
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(db_data, f, indent=2)
    except Exception as e:
        click.echo(f"      ⚠️ Failed to write early tmp-ContextDB to {db_path}: {e}")

    return {
        "relevance_score": effective_relevance,
        "domain_relevance": result.domain_relevance,
        "methodology_relevance": result.methodology_relevance,
        "match_type": match_type,
        "central_hypothesis": result.central_hypothesis,
        "reading_directive": result.reading_directive,
        "run_id": run_id,
        "context_db_path": context_db_path,
        "source_filename": state.get("source_filename", ""),
    }


def consolidated_specialist_node(state: PaperIngestionState) -> Dict[str, Any]:
    """Consolidated specialist node powered by a thinking model.

    Reads the entire paper context (methods + results + conclusions) in a single
    LLM invocation and extracts multiple distinct atomic insights. Uses the
    <|think|> token to activate the model's native reasoning pipeline.
    """
    _init_llms()
    config = get_runtime_config()
    model = specialist_llm.with_structured_output(
        AtomicInsightList, method="json_schema"
    )

    source_doc = state["source_doc"]
    methods_text = source_doc.methods or ""
    results_text = source_doc.results_discussion or ""
    conclusions_text = source_doc.conclusions or ""
    directive = state["reading_directive"]
    hypothesis = state.get("central_hypothesis", "")
    match_type = state.get("match_type", "full")
    feedback = state.get("feedback")

    # Select the consolidated specialist prompt
    system_instruction = config.prompts.specialist_consolidated
    system_instruction = system_instruction.format(
        directive=directive, hypothesis=hypothesis
    )

    # Prepend the <|think|> token to activate thinking mode
    system_instruction = "<|think|>\n" + system_instruction

    # Append match-type specific focus instructions
    if match_type == "methodology_only":
        system_instruction += (
            "\n\n### Focus Mode: METHODOLOGY ONLY\n"
            "Focus strictly on verifying the computational/experimental framework parameters. "
            "Ignore domain-specific chemical properties and results metrics."
        )
    elif match_type == "domain_only":
        system_instruction += (
            "\n\n### Focus Mode: DOMAIN ONLY\n"
            "Focus on results and metrics. "
            "Bypass heavy methodological framework details."
        )
    else:
        system_instruction += (
            "\n\n### Focus Mode: FULL EXTRACTION\n"
            "Extract both technical parameters and domain research data."
        )

    # Append AtomicInsightList schema guidelines
    system_instruction += (
        "\n\nYou MUST return a JSON object with a single key 'insights'. "
        "The value of 'insights' must be a list of insight objects. "
        "Each insight object must contain the following keys:\n"
        "- title: A single, standalone declarative statement (60-100 characters)\n"
        "- core_insight: Exactly 2-3 sentences using explicit nouns (200-350 characters, MUST not exceed 350 characters)\n"
        "- context_evidence: A single plain-text string with bulleted source citations and evidence "
        "(do NOT return a nested JSON object or dictionary)\n"
        "- related_vectors: A list of Obsidian WikiLinks for vault topology.\n"
        "- data_points: A flat dictionary of key details, parameters, yields, or protocols (keys and values must be strings or numbers).\n"
        "- source_quotes: A list of verbatim quotes from the source text that support the insight.\n"
        "\n\nCRITICAL: You must produce AT LEAST ONE atomic insight. "
        "Extract MULTIPLE distinct discoveries as separate insights rather than combining them."
    )

    # Include feedback from reflection retry loop if present
    feedback_text = ""
    if feedback:
        if isinstance(feedback, dict):
            feedback_parts = [f"{k}: {v}" for k, v in feedback.items() if v]
            feedback_text = "; ".join(feedback_parts)
        else:
            feedback_text = str(feedback)

    if feedback_text:
        system_instruction += (
            f"\n\nNote: You previously failed fact-checking. "
            f"Correct your output based on this feedback: {feedback_text}"
        )

    # Compose full paper context
    paper_context_parts = []
    if methods_text:
        paper_context_parts.append(f"## Methods\n{methods_text}")
    if results_text:
        paper_context_parts.append(f"## Results and Discussion\n{results_text}")
    if conclusions_text:
        paper_context_parts.append(f"## Conclusions\n{conclusions_text}")
    paper_context = "\n\n".join(paper_context_parts)

    prompt = [
        SystemMessage(content=system_instruction),
        HumanMessage(
            content=(
                f"Source Doc Title: {source_doc.title}\n\n"
                f"Paper Context:\n{paper_context}"
            )
        ),
    ]

    # Trace log
    action = "Re-analysing" if feedback_text else "Analysing"

    spec_insights = None
    with AnalysisSpinner(f"{action} full paper ({match_type} mode)..."):
        for attempt in range(1, 3):
            try:
                spec_insights = model.invoke(prompt)
                if spec_insights and spec_insights.insights:
                    break
                else:
                    raise ValueError("No insights returned by the model.")
            except Exception as e:
                log_event(
                    f"      ⚠️ Attempt {attempt} for consolidated synthesis failed: {str(e)}"
                )
                if attempt == 2:
                    raise ValueError(
                        f"LLM failed to satisfy AtomicInsightList constraints after 2 attempts. Error: {str(e)}"
                    )

    # Build raw_extractions for context DB and reflection cross-reference
    raw_extractions = []
    atomic_insights = []
    if spec_insights and spec_insights.insights:
        for ins in spec_insights.insights:
            raw_extractions.append(
                {
                    "specialist_type": "consolidated",
                    "data_points": {},
                    "source_quotes": [],
                    "title": ins.title,
                    "core_insight": ins.core_insight,
                }
            )
            atomic_insights.append(ins)

    return {
        "raw_extractions": raw_extractions,
        "atomic_insights": atomic_insights,
    }


def format_source_block(
    source_filename: str,
    evidence: str,
    data_points: Dict[str, Any],
    source_quotes: List[str],
    verified: bool = True,
    feedback: str = "",
) -> str:
    filename_stem = Path(source_filename).stem if source_filename else "unknown"
    header_suffix = " ⚠️ (Unverified)" if not verified else ""
    block = f"### Source: [[{filename_stem}]]{header_suffix}\n\n"

    if not verified and feedback:
        block += f"> **Discrepancy**: {feedback}\n\n"

    block += "**LLM-generated evidence**:\n"
    evidence_lines = [
        line.strip() for line in evidence.strip().splitlines() if line.strip()
    ]
    for line in evidence_lines:
        if line.startswith("*") or line.startswith("-"):
            block += f"{line}\n"
        else:
            block += f"* {line}\n"
    block += "\n"

    block += "**Structured technical parameters**:\n"
    if data_points:
        for k, v in data_points.items():
            block += f"* **{k}**: {v}\n"
    else:
        block += "* None\n"
    block += "\n"

    block += "**Supporting quotes**:\n"
    if source_quotes:
        for q in source_quotes:
            q_stripped = q.strip().strip('"').strip("'")
            block += f'* "{q_stripped}"\n'
    else:
        block += "* None\n"

    return block


def replace_or_append_source_block(
    markdown_content: str, source_filename: str, new_source_block: str
) -> str:
    filename_stem = Path(source_filename).stem if source_filename else "unknown"
    escaped_stem = re.escape(filename_stem)
    pattern = rf"(### Source:\s*\[\[{escaped_stem}\]\](?:\s*⚠️\s*\(Unverified\))?)(.*?)(?=\n### Source:\s*\[\[|\Z)"

    match = re.search(pattern, markdown_content, re.DOTALL | re.IGNORECASE)
    if match:
        start, end = match.span()
        updated_content = (
            markdown_content[:start] + new_source_block.strip() + markdown_content[end:]
        )
        return updated_content
    else:
        if "## Context & Evidence" not in markdown_content:
            markdown_content = (
                markdown_content.rstrip() + "\n\n---\n## Context & Evidence\n"
            )
        if not markdown_content.endswith("\n"):
            markdown_content += "\n"
        updated_content = markdown_content + "\n" + new_source_block.strip() + "\n"
        return updated_content


def reflection_review_node(state: PaperIngestionState) -> Dict[str, Any]:
    """Reflection node acting as an audit-only fact-checker.

    Cross-references each proposed atomic insight's quotes and facts against
    the raw source text. Does NOT perform synthesis — that is handled by the
    consolidated specialist node.
    """
    click.echo("   🧐 Reflection Agent: Auditing specialist extractions...")

    _init_llms()
    config = get_runtime_config()
    model = specialist_llm.with_structured_output(
        ReflectionDecision, method="json_schema"
    )

    source_doc = state["source_doc"]
    methods_text = source_doc.methods or ""
    results_text = source_doc.results_discussion or ""
    conclusions_text = source_doc.conclusions or ""
    atomic_insights = state.get("atomic_insights", [])
    retry_count = state["retry_count"]

    system_instruction = config.prompts.reflection
    system_instruction += (
        "\n\nYou MUST return a JSON object containing the following keys:\n"
        "- verified: A boolean (true or false) indicating if no discrepancies or hallucinations are found\n"
        "- feedback: A detailed message describing the discrepancy if verified is false, otherwise empty\n"
        "- insight_dossier: A brief cross-reference summary if verified is true"
    )

    # Format the insights for cross-referencing
    insights_text = json.dumps(
        [
            {
                "title": ins.title if hasattr(ins, "title") else str(ins),
                "core_insight": ins.core_insight
                if hasattr(ins, "core_insight")
                else "",
                "context_evidence": ins.context_evidence
                if hasattr(ins, "context_evidence")
                else "",
            }
            for ins in atomic_insights
        ],
        indent=2,
    )

    prompt = [
        SystemMessage(content=system_instruction),
        HumanMessage(
            content=(
                f"Original Methods Text:\n{methods_text}\n\n"
                f"Original Results Text:\n{results_text}\n\n"
                f"Original Conclusions Text:\n{conclusions_text}\n\n"
                f"Proposed Atomic Insights:\n{insights_text}\n\n"
                f"Retry Count: {retry_count}"
            )
        ),
    ]

    result = model.invoke(prompt)
    max_loops = (
        config.blocks.analysis_block.max_debate_loops
        if config.blocks.analysis_block.max_debate_loops is not None
        else 3
    )

    if not result.verified:
        if retry_count < max_loops:
            click.echo(
                f"      ⚠️ Discrepancy found. Routing back to Specialist (Loop {retry_count + 1}/{max_loops})."
            )

            feedback_dict = {"consolidated": result.feedback}

            return {"feedback": feedback_dict, "retry_count": retry_count + 1}
        else:
            click.echo(
                f"      ⚠️ Debate limit ({max_loops}) reached without successful verification. "
                f"Proceeding to Network Block with unverified status."
            )

    # Write to tmp-ContextDB JSON file in quarantine folder
    from pathlib import Path

    db_path = Path(
        state.get("context_db_path")
        or (
            Path(config.system.quarantine_path)
            / f"tmp-context-{state.get('run_id')}.json"
        )
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)

    insights_db = {}
    for idx, ins in enumerate(atomic_insights):
        insights_db[f"insight_{idx + 1}"] = {
            "title": ins.title if hasattr(ins, "title") else str(ins),
            "core_insight": ins.core_insight if hasattr(ins, "core_insight") else "",
        }

    # Load existing context DB data if available, to merge instead of overwrite
    db_data = {}
    if db_path.exists():
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                db_data = json.load(f)
        except Exception as e:
            click.echo(
                f"      ⚠️ Failed to read existing tmp-ContextDB at {db_path}: {e}"
            )

    db_data.update(
        {
            "insights": insights_db,
            "verified": result.verified,
            "feedback": result.feedback if not result.verified else "",
            "insight_dossier": result.insight_dossier,
            "status": "reviewed",
        }
    )

    try:
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(db_data, f, indent=2)
    except Exception as e:
        click.echo(f"      ⚠️ Failed to write tmp-ContextDB to {db_path}: {e}")

    # Format context_evidence blocks for each atomic insight
    for ins in atomic_insights:
        ins.context_evidence = format_source_block(
            source_filename=state.get("source_filename", ""),
            evidence=ins.context_evidence if hasattr(ins, "context_evidence") else "",
            data_points=ins.data_points if hasattr(ins, "data_points") else {},
            source_quotes=ins.source_quotes if hasattr(ins, "source_quotes") else [],
            verified=result.verified,
            feedback=result.feedback if not result.verified else "",
        )

    return {
        "insight_dossier": result.insight_dossier,
        "atomic_insights": atomic_insights,
        "feedback": {"discrepancy": result.feedback} if not result.verified else None,
    }


def supervisor_network_node(state: PaperIngestionState) -> Dict[str, Any]:
    """Supervisor network node with in-flight deduplication.

    Compares new insights against an in-memory queue to merge duplicates,
    then queries local LanceDB for related vault context and determines
    whether to append to an existing note, contradict it, or create a new note.
    """
    config = get_runtime_config()
    threshold = config.blocks.supervisor_block.relevance_threshold or 0.75

    if state.get("relevance_score", 0.0) < threshold:
        stream_supervisor_thought(
            "",
            "Relevance triage score below threshold. Bypassing analysis block and network insertion.",
        )
        return {
            "vault_context": [],
            "final_markdown": "",
            "routing_results": [
                {
                    "decision": "bypass",
                    "target_note": "",
                    "note_title": "",
                    "markdown_content": "",
                    "insight": None,
                }
            ],
            "in_flight_notes": [],
        }

    # Load verification status from tmp-ContextDB JSON file
    from pathlib import Path

    db_path = Path(
        state.get("context_db_path")
        or (
            Path(config.system.quarantine_path)
            / f"tmp-context-{state.get('run_id')}.json"
        )
    )

    verified = True
    discrepancy_feedback = ""

    if db_path.exists():
        try:
            with open(db_path, "r", encoding="utf-8") as f:
                db_data = json.load(f)
                verified = db_data.get("verified", True)
                discrepancy_feedback = db_data.get("feedback", "")
        except Exception as e:
            click.echo(f"      ⚠️ Failed to read tmp-ContextDB at {db_path}: {e}")

    stream_supervisor_thought(
        config.blocks.supervisor_block.model, "Querying Librarian Tool (LanceDB)..."
    )

    from stratasift.tools.librarian import LanceLibrarian

    vault_path = config.system.get_expanded_vault_path()
    librarian = LanceLibrarian(vault_path)

    # In-flight deduplication setup
    embedding_fn = SimpleEmbeddingFunction()
    in_flight_notes = list(state.get("in_flight_notes", []))

    routing_results = []
    note_contents = {}  # Tracks in-memory markdown content for target notes
    vault_context_titles = []

    relevance_threshold = config.blocks.supervisor_block.relevance_threshold or 0.75

    for insight in state.get("atomic_insights", []):
        search_block = insight.search_block

        # In-flight deduplication: check against already-processed insights this run
        is_duplicate = False
        insight_vector = embedding_fn([search_block])[0]

        for existing_note in in_flight_notes:
            existing_vector = existing_note.get("vector", [])
            if existing_vector:
                similarity = cosine_similarity(insight_vector, existing_vector)
                if similarity >= relevance_threshold:
                    stream_supervisor_thought(
                        config.blocks.supervisor_block.model,
                        f"Identified contradiction. Proposing append to '{target_note}' with discrepancy log.",
                    )
                    # Merge context_evidence into the existing note
                    existing_note["merged_evidence"] = (
                        existing_note.get("merged_evidence", "")
                        + "\n\n"
                        + insight.context_evidence
                    )
                    is_duplicate = True
                    break

        if is_duplicate:
            continue

        # Track this insight in the in-flight queue
        in_flight_notes.append(
            {
                "title": insight.title,
                "search_block": search_block,
                "vector": insight_vector,
                "merged_evidence": "",
            }
        )

        matches = librarian.search_vault(search_block, n_results=3)
        similarity = matches[0].get("similarity", 1.0) if matches else 0.0

        if matches:
            click.echo(
                f'      🔗 Found {len(matches)} related note: "{matches[0]["title"]}.md" (Similarity: {similarity:.2f})'
            )

        if matches and similarity >= relevance_threshold:
            decision = "append"
            target_note = matches[0]["id"]
            if matches[0]["title"] not in vault_context_titles:
                vault_context_titles.append(matches[0]["title"])
            stream_supervisor_thought(
                config.blocks.supervisor_block.model,
                f"Synthesising final note for insight '{insight.title}'...",
            )
        else:
            decision = "create"
            target_note = ""
            stream_supervisor_thought(
                config.blocks.supervisor_block.model,
                f"Routing insight '{insight.title}' to Create New action"
                + (
                    f" (Score {similarity:.2f} < {relevance_threshold})"
                    if matches
                    else " (Database empty)"
                ),
            )

        note_title = insight.title

        if decision == "append":
            if target_note in note_contents:
                existing_content = note_contents[target_note]
            else:
                target_note_path = vault_path / target_note
                if target_note_path.exists():
                    with open(target_note_path, "r", encoding="utf-8") as f:
                        existing_content = f.read()
                else:
                    existing_content = matches[0].get("content", "")

            updated_content = replace_or_append_source_block(
                markdown_content=existing_content,
                source_filename=state.get("source_filename", ""),
                new_source_block=insight.context_evidence,
            )

            if not verified:
                updated_content = updated_content.replace(
                    'status: "verified"', 'status: "unverified"'
                )
                if "## Discrepancy Log" not in updated_content:
                    updated_content += (
                        f"\n\n---\n## Discrepancy Log\n> ⚠️ {discrepancy_feedback}\n"
                    )

            note_contents[target_note] = updated_content
            final_markdown_for_this_insight = updated_content

        else:
            dummy_id = generate_insight_id()
            date_str = datetime.date.today().strftime("%Y-%m-%d")
            new_content = format_insight(insight, dummy_id, date_str, ["insight"])

            if not verified:
                new_content = new_content.replace(
                    'status: "verified"', 'status: "unverified"'
                )
                new_content += (
                    f"\n\n---\n## Discrepancy Log\n> ⚠️ {discrepancy_feedback}\n"
                )

            from stratasift.utils.file_io import sanitise_filename

            target_note_created = f"{sanitise_filename(note_title)}.md"
            note_contents[target_note_created] = new_content
            final_markdown_for_this_insight = new_content

        routing_results.append(
            {
                "decision": decision,
                "target_note": target_note,
                "note_title": note_title,
                "markdown_content": final_markdown_for_this_insight,
                "insight": insight,
            }
        )

    final_markdown = "\n\n".join(r["markdown_content"] for r in routing_results)

    return {
        "vault_context": vault_context_titles,
        "final_markdown": final_markdown,
        "routing_results": routing_results,
        "in_flight_notes": in_flight_notes,
    }
