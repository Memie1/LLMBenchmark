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
Loads the models low, medium and high

NOTE FOR FUTURE JAMIE:
plot benchmark results out in matplotlib for comparisons in dissertation


checks.py:
runs basic automated checks on the model output
checks for non empty outputs, whether it mentions being an AI, or rather it follows rules

benchmark_runner.py:
ruins teh benchmarks

plot_results:
reads the resutls from the benchmarks, creates graphs for comparison