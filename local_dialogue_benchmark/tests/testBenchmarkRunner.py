from pathlib import Path

import csv

from src import benchmark_runner


def test_parse_preset_selection_supports_low_medium_aliases():
    assert benchmark_runner.parse_preset_selection("low,medium") == ["low", "medium"]
    assert benchmark_runner.parse_preset_selection("both") == ["low", "medium"]


def test_parse_preset_selection_supports_all_and_deduplicates():
    assert benchmark_runner.parse_preset_selection("all") == ["low", "medium", "high"]
    assert benchmark_runner.parse_preset_selection("low,medium,low") == ["low", "medium"]


def test_run_benchmark_writes_rows_for_multiple_presets(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_file = results_dir / "benchmark_results.csv"
    scored_results_file = results_dir / "scored_results.csv"
    plots_dir = results_dir / "plots"

    monkeypatch.setattr(benchmark_runner, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(benchmark_runner, "RESULTS_FILE", results_file)
    monkeypatch.setattr(benchmark_runner, "SCORED_RESULTS_FILE", scored_results_file)
    monkeypatch.setattr(benchmark_runner, "PLOTS_DIR", plots_dir)
    monkeypatch.setattr(
        benchmark_runner,
        "load_scenarios",
        lambda _: [
            {
                "title": "Scenario A",
                "character": "You are a shopkeeper.",
                "hard_rules": ["Stay in character."],
                "turns": ["hello"],
            }
        ],
    )

    def fake_discover_model_files(preset: str) -> list[Path]:
        return [Path(f"{preset}-model.gguf")]

    monkeypatch.setattr(benchmark_runner, "discover_model_files", fake_discover_model_files)
    monkeypatch.setattr(
        benchmark_runner,
        "load_model",
        lambda model_path, preset: {"preset": preset, "model_path": str(model_path)},
    )
    monkeypatch.setattr(
        benchmark_runner,
        "generate_reply",
        lambda model, messages: f"reply-{model['preset']}",
    )
    monkeypatch.setattr(
        benchmark_runner,
        "evaluate_reply",
        lambda scenario, reply: {
            "word_count": 1,
            "passed_basic_checks": True,
            "failure_reason": "",
            "mentions_ai": False,
            "under_30_words": True,
        },
    )

    benchmark_runner.run_benchmark("low,medium")

    with open(results_file, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert [row["preset"] for row in rows] == ["low", "medium"]
    assert [row["model_file"] for row in rows] == ["low-model.gguf", "medium-model.gguf"]