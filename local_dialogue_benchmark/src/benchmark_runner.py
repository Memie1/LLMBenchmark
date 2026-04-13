from __future__ import annotations

import csv
import time
import argparse
import gc
from pathlib import Path

from src.prompts import load_scenarios, build_messages
from src.models import discover_model_files, load_model, generate_reply
from src.checks import evaluate_reply

# define file paths and constants 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_FILE = RESULTS_DIR / "benchmark_results.csv"
SCORED_RESULTS_FILE = RESULTS_DIR / "scored_results.csv"
PLOTS_DIR = RESULTS_DIR / "plots"

# these are the columns saved for each benchmark turn
RESULTS_COLUMNS = [
    "timestamp",
    "preset",
    "model_file",
    "scenario_title",
    "user_input",
    "raw_output",
    "response_time_ms",
    "word_count",
    "passed_basic_checks",
    "failure_reason",
    "mentions_ai",
    "under_30_words",
]

# remove old results so a fresh run does not mix with older benchmark data
def clear_previous_outputs() -> bool:
    cleared_anything = False

    for path in (RESULTS_FILE, SCORED_RESULTS_FILE):
        if path.exists():
            path.unlink()
            cleared_anything = True

    if PLOTS_DIR.exists():
        for plot_file in PLOTS_DIR.glob("*.png"):
            plot_file.unlink()
            cleared_anything = True

    return cleared_anything


def ensure_results_file():
    # make sure the results folder exists before trying to write anything
    RESULTS_DIR.mkdir(exist_ok=True)

    if not RESULTS_FILE.exists():
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(RESULTS_COLUMNS)
        return

    with open(RESULTS_FILE, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    if not rows:
        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(RESULTS_COLUMNS)
        return

    header = rows[0]
    if header == RESULTS_COLUMNS:
        return

    # support older csv files that were created before model_file was added
    if header == [
        "timestamp",
        "preset",
        "scenario_title",
        "user_input",
        "raw_output",
        "response_time_ms",
        "word_count",
        "passed_basic_checks",
        "failure_reason",
        "mentions_ai",
        "under_30_words",
    ]:
        migrated_rows = [RESULTS_COLUMNS]
        for row in rows[1:]:
            migrated_rows.append(row[:2] + [""] + row[2:])

        with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(migrated_rows)
        return

    raise ValueError(
        f"Unexpected results header in '{RESULTS_FILE}'. Please inspect the file before rerunning."
    )


def run_benchmark(preset: str, append: bool = False):
    # load the current scenarios and find all models inside the chosen preset folder
    scenarios = load_scenarios(PROJECT_ROOT / "scenarios.json")
    model_files = discover_model_files(preset)

    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Discovered {len(model_files)} model(s) in preset '{preset}'")

    if not append and clear_previous_outputs():
        print("Cleared previous benchmark results, scored results, and plot images")

    ensure_results_file()

    # append each generated reply as a row so scoring and plotting can process it later
    with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        for model_path in model_files:
            # load one model at a time to keep memory usage predictable
            print(f"Loading model: {model_path.name} ({preset})...")
            model = load_model(model_path, preset)
            print("Model loaded\n")

            for scenario in scenarios:
                print(f"Running {model_path.name}: {scenario['title']}")

                turns = scenario.get("turns", [])
                if not turns:
                    # Support legacy single-turn format
                    turns = [scenario.get("user_input", "Hello")]

                # store the conversation so later turns can include earlier context
                dialogue_history = []

                for i, user_input in enumerate(turns):
                    print(f"Turn {i+1}/{len(turns)}: {user_input}")
                    messages = build_messages(scenario, dialogue_history, user_input)

                    # time only the model generation call because that is what we want to compare
                    start = time.perf_counter()
                    reply = generate_reply(model, messages)
                    elapsed = (time.perf_counter() - start) * 1000

                    # basic checks run immediately so the raw csv stores both output and quick validation
                    result = evaluate_reply(scenario, reply)

                    writer.writerow([
                        time.strftime("%Y-%m-%dT%H:%M:%S"),
                        preset,
                        model_path.name,
                        scenario["title"],
                        user_input,
                        reply,
                        round(elapsed, 2),
                        result["word_count"],
                        result["passed_basic_checks"],
                        result["failure_reason"],
                        result["mentions_ai"],
                        result["under_30_words"],
                    ])

                    print(f"Reply: {reply}")
                    print(f"Time: {round(elapsed, 2)} ms")
                    print(f"Pass: {result['passed_basic_checks']}")

                    # Update history for next turn
                    dialogue_history.append({"user": user_input, "assistant": reply})

                print("-" * 40)

            # free the current model before loading the next one
            del model
            gc.collect()


if __name__ == "__main__":
    # small cli so the same runner can be used for low / medium / high presets
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="low", help="low / medium / high")
    parser.add_argument(
        "--append",
        action="store_true",
        help="append to existing benchmark results instead of clearing previous outputs",
    )

    args = parser.parse_args()

    run_benchmark(args.preset, append=args.append)