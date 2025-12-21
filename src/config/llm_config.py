"""
Centralized LLM configuration for the ANeuk AI project.

This module loads LLM configurations from llm_config.yaml and provides
factory functions to create LLM instances for different purposes.

All model configurations are defined in llm_config.yaml.
No environment variables are used for model selection.

Updated for LangChain v1.2+ with init_chat_model for unified model initialization.
Supports Gemini thinking_level and OpenAI reasoning_effort parameters.
"""

from pathlib import Path
from typing import List, Dict, Any
from omegaconf import OmegaConf, DictConfig
from langchain.chat_models import init_chat_model


# Load configuration from YAML
CONFIG_PATH = Path(__file__).parent / "llm_config.yaml"
_config = OmegaConf.load(CONFIG_PATH)


def _get_model_provider(model_name: str) -> str:
    """
    Determines the model provider based on the model name.

    Args:
        model_name: The name of the model.

    Returns:
        Provider string for init_chat_model.
    """
    model_lower = model_name.lower()

    if "gpt" in model_lower or "o3-mini" in model_name or "o4-mini" in model_name:
        return "openai"
    elif "gemini" in model_lower:
        return "google_genai"
    elif "claude" in model_lower:
        return "anthropic"
    else:
        raise NotImplementedError(f"LLM {model_name} provider not supported")


def _get_single_llm_model(model_cfg: DictConfig):
    """
    Creates a single LLM instance from a model configuration using init_chat_model.

    Args:
        model_cfg: Configuration dict containing model_name, temperature, etc.

    Returns:
        LLM instance initialized via init_chat_model (provider-agnostic)
    """
    model_name = model_cfg.model_name
    provider = _get_model_provider(model_name)

    # Build kwargs for init_chat_model
    kwargs = {
        "model": model_name,
        "model_provider": provider,
    }

    # Common parameters
    if "temperature" in model_cfg:
        kwargs["temperature"] = model_cfg.temperature
    if "max_tokens" in model_cfg:
        kwargs["max_tokens"] = model_cfg.max_tokens

    # OpenAI-specific parameters
    if provider == "openai":
        if "reasoning_effort" in model_cfg:
            kwargs["reasoning_effort"] = model_cfg.reasoning_effort

    # Google Gemini-specific parameters (langchain-google-genai native support)
    if provider == "google_genai":
        if "thinking_level" in model_cfg:
            kwargs["thinking_level"] = model_cfg.thinking_level
        if "thinking_budget" in model_cfg:
            kwargs["thinking_budget"] = model_cfg.thinking_budget

    return init_chat_model(**kwargs)


def get_llm_models(cfg_section: DictConfig) -> Dict[str, Any]:
    """
    Creates a dictionary of LLM instances from a configuration section.

    Args:
        cfg_section: OmegaConf DictConfig where each key is an LLM name
                    and value contains its configuration.

    Returns:
        Dictionary mapping LLM names to initialized LLM objects.
    """
    llms = {}
    for llm_name, model_cfg in cfg_section.items():
        if isinstance(model_cfg, DictConfig) and "model_name" in model_cfg:
            llms[llm_name] = _get_single_llm_model(model_cfg)
    return llms


# Factory functions for specific tasks
def get_chat_llm():
    """
    Get LLM for chat generation.

    Configuration: llm_config.yaml -> models.chat
    """
    return _get_single_llm_model(_config.models.chat)


def get_chat_fallback_llm():
    """
    Get fallback LLM for chat generation when primary model fails.

    Configuration: llm_config.yaml -> models.chat_fallback
    """
    return _get_single_llm_model(_config.models.chat_fallback)


def get_diary_llm():
    """
    Get LLM for diary generation.

    Configuration: llm_config.yaml -> models.diary
    """
    return _get_single_llm_model(_config.models.diary)


def get_diary_split_llm():
    """
    Get LLM for diary paragraph splitting.

    Configuration: llm_config.yaml -> models.diary_split
    """
    return _get_single_llm_model(_config.models.diary_split)


def get_emotion_finding_llms() -> List:
    """
    Get LLMs for emotion finding with varying temperatures.

    Returns a list of 4 LLM instances with different temperatures
    for diverse emotion extraction.

    Configuration: llm_config.yaml -> models.emotion_finding
    """
    emotion_cfg = _config.models.emotion_finding
    return [
        _get_single_llm_model(emotion_cfg.primary),
        _get_single_llm_model(emotion_cfg.low),
        _get_single_llm_model(emotion_cfg.medium),
        _get_single_llm_model(emotion_cfg.high),
    ]


def get_remake_llm():
    """
    Get LLM for sentence remaking.

    Configuration: llm_config.yaml -> models.remake
    """
    return _get_single_llm_model(_config.models.remake)


def _format_model_info(cfg: DictConfig) -> str:
    """Format model configuration for logging."""
    parts = [cfg.model_name]
    if "temperature" in cfg:
        parts.append(f"temp={cfg.temperature}")
    if "thinking_level" in cfg:
        parts.append(f"thinking={cfg.thinking_level}")
    if "thinking_budget" in cfg:
        parts.append(f"thinking_budget={cfg.thinking_budget}")
    if "reasoning_effort" in cfg:
        parts.append(f"reasoning={cfg.reasoning_effort}")
    return f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0]


# Logging configuration on module load
print("=" * 60)
print("LLM Configuration Loaded from: llm_config.yaml")
print("=" * 60)
print(f"Chat          : {_format_model_info(_config.models.chat)}")
print(f"Chat Fallback : {_format_model_info(_config.models.chat_fallback)}")
print(f"Diary         : {_format_model_info(_config.models.diary)}")
print(f"Diary Split   : {_format_model_info(_config.models.diary_split)}")
print(f"Emotion (x4)  : {_format_model_info(_config.models.emotion_finding.primary)}")
print(f"Remake        : {_format_model_info(_config.models.remake)}")
print("=" * 60)
