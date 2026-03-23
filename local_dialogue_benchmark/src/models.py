
from dataclasses import dataclass
# Llama from: https://github.com/abetlen/llama-cpp-python
from llama_cpp import Llama

@dataclass
class ModelConfig: 
    name: str
    model_path: str
    n_ctx: int = 2048
    n_threads: int = 8
    n_gpu_layers: int = 0


MODEL_PRESETS = {
    "low": ModelConfig(
        name="low",
        model_path="models/low.gguf",
        n_ctx=1024,
        n_threads=6,
        n_gpu_layers=0,
    ),
    "medium": ModelConfig(
        name="medium",
        model_path="models/medium.gguf",
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=0,
    ),
    "high": ModelConfig(
        name="high",
        model_path="models/high.gguf",
        n_ctx=4096,
        n_threads=12,
        n_gpu_layers=0,
    )
}

# Function to load the model based on the preset
# i guess this will be later defined in the benchmarking script
def load_model(preset: str) -> Llama:
    if preset not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {preset}")
    
    config = MODEL_PRESETS[preset]

    model = Llama(
        model_path=config.model_path,
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
    temperature: float = 0.0) -> str:

    response = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response["choices"][0]["message"]["content"].strip()