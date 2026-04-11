
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llama_cpp import Llama

@dataclass
class ModelConfig: 
    name: str
    directory_name: str
    n_ctx: int = 2048
    n_threads: int = 8
    n_gpu_layers: int = 0


MODEL_PRESETS = {
    "low": ModelConfig(
        name="low",
        directory_name="low",
        n_ctx=1024,
        n_threads=6,
        n_gpu_layers=0,
    ),
    "medium": ModelConfig(
        name="medium",
        directory_name="medium",
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=0,
    ),
    "high": ModelConfig(
        name="high",
        directory_name="high",
        n_ctx=4096,
        n_threads=12,
        n_gpu_layers=0,
    )
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def get_preset_config(preset: str) -> ModelConfig:
    normalized_preset = preset.lower()

    if normalized_preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")

    return MODEL_PRESETS[normalized_preset]


def resolve_preset_directory(preset: str) -> Path:
    config = get_preset_config(preset)
    expected_dir = MODELS_DIR / config.directory_name

    if expected_dir.exists() and expected_dir.is_dir():
        return expected_dir

    for child in MODELS_DIR.iterdir():
        if child.is_dir() and child.name.lower() == config.directory_name.lower():
            return child

    raise FileNotFoundError(
        f"Preset directory not found for '{preset}'. Expected '{expected_dir}'."
    )


def discover_model_files(preset: str) -> list[Path]:
    preset_dir = resolve_preset_directory(preset)
    model_files = sorted(
        path for path in preset_dir.iterdir() if path.is_file() and path.suffix.lower() == ".gguf"
    )

    if not model_files:
        raise FileNotFoundError(f"No .gguf files found in preset directory '{preset_dir}'.")

    return model_files


def load_model(model_path: str | Path, preset: str) -> Llama:
    from llama_cpp import Llama

    config = get_preset_config(preset)
    resolved_model_path = Path(model_path)

    model = Llama(
        model_path=str(resolved_model_path),
        n_ctx=config.n_ctx,
        n_threads=config.n_threads,
        n_gpu_layers=config.n_gpu_layers,
        verbose=False
    )
    return model


def generate_reply(
    model: Llama,
    messages: list[dict],
    max_tokens: int = 128,
    temperature: float = 0.0) -> str: # temperature is set to 0, ideal for benchmarking but not for real use
    

    response = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"].strip()