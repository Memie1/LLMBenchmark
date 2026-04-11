Project: Benchmarking Ai that can be ran on consumer hardware for the future of game development and simulations


Scenarios.json: 
A public dataset for benchmarking ai

title: Title of role
character: Character descriptions
hard_rules: self explanitory
user_input: inputs for what the user could say, testing different scenarios



prompts.py:
1) loads the scenarios
2) builds the prompt cleanly


models.py:
Finds compatible GGUF models inside the selected preset folder and loads them one by one

NOTE FOR FUTURE JAMIE:
plot benchmark results out in matplotlib for comparisons in dissertation


checks.py:
runs basic automated checks on the model output
checks for non empty outputs, whether it mentions being an AI, or rather it follows rules

benchmark_runner.py:
runs the benchmarks

plot_results:
reads the resutls from the benchmarks, creates graphs for comparison



================================

Models selected:
going for these as a benchmark for vram
due to the size of files, I am unable to upload them to the github


4gb = low
8gb = medium
16gb = high

Low models:
llama-3.2-3b-instruct-gguf: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF


Model folders:
Place compatible GGUF files inside preset folders under models/.
Example layout:

models/
- low/
- medium/
- high/

The benchmark runner enumerates every .gguf file inside the chosen preset folder and benchmarks them sequentially.