from __future__ import annotations

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "results" / "scored_results.csv"
SCENARIOS_FILE = PROJECT_ROOT / "scenarios.json"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def normalize_title(text: str) -> str:
    # normalize titles so matching still works even if spacing/case differs
    return " ".join(text.strip().lower().split())


def load_memory_scenarios() -> set[str]:
    # read memory scenario titles so we can build a memory-only chart later
    if not SCENARIOS_FILE.exists():
        return set()

    with open(SCENARIOS_FILE, "r", encoding="utf-8") as handle:
        scenarios = json.load(handle)

    return {
        normalize_title(scenario["title"])
        for scenario in scenarios
        if scenario.get("type") == "memory"
    }


def main():
    if not RESULTS_FILE.exists():
        print("No scored results file found. Run benchmark_runner and scoring first.")
        return

    # clear old images first so the plots folder only shows the latest run
    PLOTS_DIR.mkdir(exist_ok=True)
    for old_plot in PLOTS_DIR.glob("*.png"):
        old_plot.unlink()

    # load the scored csv because the useful charts now depend on final_score
    df = pd.read_csv(RESULTS_FILE)
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce")
    df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce")
    df["memory_score"] = pd.to_numeric(df["memory_score"], errors="coerce")

    # drop rows that are missing the values needed for the main comparisons
    df = df.dropna(subset=["response_time_ms", "final_score"])

    # 1. Average response time per model
    plt.figure()
    df.groupby("model_file")["response_time_ms"].mean().sort_values().plot(kind="bar")
    plt.title("Average Response Time per Model")
    plt.ylabel("ms")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_response_time_per_model.png")
    plt.close()

    # 2. Average final score per scenario
    plt.figure()
    df.groupby("scenario_title")["final_score"].mean().sort_values(ascending=False).plot(kind="bar")
    plt.title("Average Final Score per Scenario")
    plt.ylabel("Final Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_final_score_per_scenario.png")
    plt.close()

    # 3. Average final score per model
    plt.figure()
    df.groupby("model_file")["final_score"].mean().sort_values(ascending=False).plot(kind="bar")
    plt.title("Average Final Score per Model")
    plt.ylabel("Final Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "avg_final_score_per_model.png")
    plt.close()

    # 4. Speed vs average final score by model
    summary = df.groupby("model_file").agg({
        "response_time_ms": "mean",
        "final_score": "mean"
    })

    plt.figure()
    plt.scatter(summary["response_time_ms"], summary["final_score"])

    for model_file, row in summary.iterrows():
        plt.text(row["response_time_ms"], row["final_score"], model_file)

    plt.xlabel("Avg Response Time (ms)")
    plt.ylabel("Average Final Score")
    plt.title("Speed vs Final Score by Model")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "speed_vs_final_score_by_model.png")
    plt.close()

    # 5. Memory score per memory scenario so memory-specific tests can be inspected separately
    memory_titles = load_memory_scenarios()
    if memory_titles:
        memory_df = df[df["scenario_title"].map(normalize_title).isin(memory_titles)]
        if not memory_df.empty:
            plt.figure()
            memory_df.groupby("scenario_title")["memory_score"].mean().sort_values(ascending=False).plot(kind="bar")
            plt.title("Average Memory Score per Memory Scenario")
            plt.ylabel("Memory Score")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "avg_memory_score_per_memory_scenario.png")
            plt.close()

    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()