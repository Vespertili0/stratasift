import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# Load environment variables from .env file
load_dotenv()


class PromptsConfig(BaseModel):
    triage: str = (
        "You are the Principal Investigator (Supervisor Agent). "
        "Evaluate the abstract/introduction and conclusions of the ingested paper against the user's domain interests and methodology interests. "
        "Output two relevance scores: domain_relevance (0.0 to 1.0) and methodology_relevance (0.0 to 1.0). "
        "Formulate a central_hypothesis representing the paper's core scientific contribution. "
        "If max(domain_relevance, methodology_relevance) >= 0.75, generate a reading_directive. "
        "Otherwise, leave reading_directive empty."
    )
    specialist_consolidated: str = (
        "You are a scientific Specialist agent performing holistic analysis. "
        "Your task is to read the full paper context (methods, results, and conclusions) "
        "and extract distinct atomic insights along with their supportive and relevant details, "
        "specifically the key technical parameters and their verbatim source quotes"
        "according to the reading directive: '{directive}'. "
        "Central hypothesis under evaluation: '{hypothesis}'."
    )
    reflection: str = (
        "You are a Senior Researcher acting as the Reflection and Fact-Checking Agent. "
        "Cross-reference each proposed atomic insight's quotes and facts against the raw source text. "
        "Identify any discrepancies, incorrect metrics, conflated contexts, or hallucinations. "
        "If verified, produce a brief cross-reference summary as insight_dossier. "
        "If any discrepancy is found, set verified=False and write a detailed feedback message."
    )
    synthesis: str = (
        "You are a Senior Researcher. Synthesise the verified scientific facts into an AtomicInsight.\n"
        "Constraints:\n"
        "- Title: 60-100 characters. Single standalone declarative statement.\n"
        "- Core Insight: 200-350 characters. 2-3 sentences using explicit nouns.\n"
        "- Context & Evidence: The source citations and bullet points.\n"
        "- Related Vectors: WikiLinks for vault topology."
    )


class SystemConfig(BaseModel):
    obsidian_vault_path: str
    quarantine_path: str
    domain_interests: List[str] = Field(
        default_factory=lambda: ["general scientific literature"],
        description="List of domain/subject-matter research interests for triage scoring.",
    )
    methodology_interests: List[str] = Field(
        default_factory=lambda: ["general research methodology"],
        description="List of computational/experimental methodology interests for triage scoring.",
    )

    def get_expanded_vault_path(self) -> Path:
        """Expand path helper (e.g. expanding ~ to home directory)."""
        expanded = os.path.expanduser(self.obsidian_vault_path)
        return Path(expanded).resolve()


class BlockConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    relevance_threshold: Optional[float] = None
    max_debate_loops: Optional[int] = None


class BlocksConfig(BaseModel):
    supervisor_block: BlockConfig
    analysis_block: BlockConfig


class GeminiConfig(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    api_key: Optional[str] = None


class OllamaLocalConfig(BaseModel):
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None


class OllamaCloudConfig(BaseModel):
    model: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = None
    api_key: Optional[str] = None


class ProvidersConfig(BaseModel):
    gemini: Optional[GeminiConfig] = None
    ollama_local: Optional[OllamaLocalConfig] = None
    ollama_cloud: Optional[OllamaCloudConfig] = None


class EvaluationConfig(BaseModel):
    threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    model: str = Field(default="gemini-1.5-pro")


class AppConfig(BaseModel):
    system: SystemConfig
    blocks: BlocksConfig
    providers: ProvidersConfig
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    api_key_file: Optional[str] = None

    def _validate_provider(self, active: str) -> None:
        if active == "gemini":
            cfg = self.providers.gemini
            if not cfg:
                raise ValueError("Gemini provider configuration is missing.")
            if not cfg.model:
                raise ValueError("Gemini model is not configured.")
            if not cfg.api_key:
                raise ValueError("GEMINI_API_KEY is not set.")

        elif active == "ollama_local":
            cfg = self.providers.ollama_local
            if not cfg:
                raise ValueError("Ollama Local provider configuration is missing.")
            if not cfg.model:
                raise ValueError("Ollama Local model is not configured.")
            if not cfg.base_url:
                raise ValueError("Ollama Local base_url is not configured.")

        elif active == "ollama_cloud":
            cfg = self.providers.ollama_cloud
            if not cfg:
                raise ValueError("Ollama Cloud provider configuration is missing.")
            if not cfg.model:
                raise ValueError("Ollama Cloud model is not configured.")
            if not cfg.base_url:
                raise ValueError("Ollama Cloud base_url is not configured.")
            if not cfg.api_key:
                raise ValueError(
                    "Ollama Cloud api_key (OLLAMA_CLOUD_API_KEY) is not set."
                )
        else:
            raise ValueError(f"Unknown provider: {active}")

    def validate_active_provider(self) -> None:
        """Validate that the active block providers have all required fields populated.

        This prevents crashing on startup if inactive providers are missing config values
        or environment keys (complying with FR-1).
        """
        self._validate_provider(self.blocks.supervisor_block.provider)
        self._validate_provider(self.blocks.analysis_block.provider)


def recursive_merge(d1: dict, d2: dict) -> None:
    """Recursively merge dictionary d2 into d1 in-place."""
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1 and isinstance(d1[k], dict):
            recursive_merge(d1[k], v)
        else:
            d1[k] = v


def load_config(
    config_path: str = "config.yaml", validate_active: bool = True
) -> AppConfig:
    """Load config.yaml and return a validated AppConfig instance.

    This function handles environment variable overrides and parses settings safely.
    It looks for .stratasift/config.json inside the vault path and merges it if present.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            raw_data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {config_path}: {e}")

    # First expand vault path to locate potential config.json
    system_section = raw_data.get("system", {})
    obsidian_vault_path = system_section.get("obsidian_vault_path")
    if obsidian_vault_path:
        expanded_vault = Path(os.path.expanduser(obsidian_vault_path)).resolve()
        json_config_path = expanded_vault / ".stratasift" / "config.json"

        if json_config_path.is_file():
            try:
                with open(json_config_path, "r", encoding="utf-8") as f:
                    import json

                    json_data = json.load(f)
                recursive_merge(raw_data, json_data)
            except Exception as e:
                raise ValueError(
                    f"Failed to merge vault-level configuration from {json_config_path}: {e}"
                )

    # Parse using Pydantic BaseModel
    try:
        config = AppConfig.model_validate(raw_data)
    except ValidationError as e:
        raise ValueError(f"Configuration validation failed: {e}")

    # Resolve keys from key file if specified
    gemini_key = None
    ollama_cloud_key = None
    if config.api_key_file:
        key_file_path = Path(os.path.expanduser(config.api_key_file)).resolve()
        if key_file_path.is_file():
            try:
                with open(key_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip()
                            if (v.startswith('"') and v.endswith('"')) or (
                                v.startswith("'") and v.endswith("'")
                            ):
                                v = v[1:-1]
                            if k == "GEMINI_API_KEY":
                                gemini_key = v
                            elif k == "OLLAMA_CLOUD_API_KEY":
                                ollama_cloud_key = v
            except Exception as e:
                raise ValueError(f"Failed to read api_key_file at {key_file_path}: {e}")

    # Fallback to environment variables if not resolved
    if not gemini_key:
        gemini_key = os.environ.get("GEMINI_API_KEY")
    if not ollama_cloud_key:
        ollama_cloud_key = os.environ.get("OLLAMA_CLOUD_API_KEY")

    # Inject GEMINI_API_KEY from environment if present
    if gemini_key:
        if not config.providers.gemini:
            config.providers.gemini = GeminiConfig()
        config.providers.gemini.api_key = gemini_key

    # Inject OLLAMA_CLOUD_API_KEY from environment if present
    if ollama_cloud_key:
        if not config.providers.ollama_cloud:
            config.providers.ollama_cloud = OllamaCloudConfig()
        config.providers.ollama_cloud.api_key = ollama_cloud_key

    # Soft validate active provider
    if validate_active:
        config.validate_active_provider()

    return config


# Runtime configurations state
_runtime_config: Optional[AppConfig] = None


def set_runtime_config(config: AppConfig) -> None:
    """Set the active configuration for the runtime environment."""
    global _runtime_config
    _runtime_config = config


def get_runtime_config() -> AppConfig:
    """Get the active configuration, falling back to config.yaml if unset."""
    global _runtime_config
    if _runtime_config is None:
        return load_config("config.yaml")
    return _runtime_config
