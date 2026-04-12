import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, Iterable, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_FILE = PROJECT_ROOT / "results" / "benchmark_results.csv"
SCENARIOS_FILE = PROJECT_ROOT / "scenarios.json"
OUTPUT_FILE = PROJECT_ROOT / "results" / "scored_results.csv"

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty"
}
STOPWORDS = {
    "a", "an", "and", "or", "but",
    "i", "you", "we", "they", "he", "she", "it",
    "is", "are", "was", "were", "be", "been",
    "to", "of", "in", "on", "at", "for", "with",
    "this", "that", "these", "those",
    "do", "did", "does", "have", "has", "had",
    "my", "your", "our", "their",
    "me", "him", "her", "them",
    "as", "if", "so", "just", "very", "really"
}
MEMORY_PROMPT_CUES = (
    "remember", "recall", "what is my", "what's my", "what did i", "what do i",
    "how many", "currently", "earlier", "before", "on my tab", "did i just order"
)
BANNED_AI_PHRASES = ["as an ai", "ai assistant", "language model", "artificial intelligence"]


def normalize_fact(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_scenario_key(text: str) -> str:
    normalized = text.replace("`", "'").replace("’", "'").replace("‘", "'")
    return normalize_fact(normalized)


def unique_in_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for value in values:
        normalized = normalize_fact(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def extract_salient_terms(
    texts: Iterable[str],
    top_n: int = 8,
    blocked_terms: Set[str] | None = None,
) -> List[str]:
    blocked = set(STOPWORDS)
    if blocked_terms:
        blocked |= blocked_terms

    counts: Counter[str] = Counter()
    first_seen: Dict[str, int] = {}
    position = 0

    for text in texts:
        for token in tokenize(text):
            if token in blocked:
                continue
            if len(token) < 3 and not token.isdigit():
                continue
            counts[token] += 1
            if token not in first_seen:
                first_seen[token] = position
                position += 1

    ranked_terms = sorted(
        counts.items(),
        key=lambda item: (-item[1], first_seen[item[0]], item[0]),
    )
    return [term for term, _ in ranked_terms[:top_n]]


def overlap_score(reply: str, salient_terms: List[str]) -> float:
    if not salient_terms:
        return 0.0

    reply_tokens = set(tokenize(reply))
    matches = sum(1 for term in salient_terms if term in reply_tokens)
    return matches / len(salient_terms)


def is_memory_probe(user_input: str, prior_turns: List[str]) -> bool:
    if not prior_turns:
        return False
    lowered = user_input.lower()
    return any(cue in lowered for cue in MEMORY_PROMPT_CUES)


def score_memory_reply(
    user_input: str,
    prior_user_turns: List[str],
    reply: str,
) -> float:
    if not is_memory_probe(user_input, prior_user_turns):
        return 0.0

    anchors = extract_salient_terms(prior_user_turns, top_n=6)
    raw_score = overlap_score(reply, anchors)

    if raw_score >= 0.5:
        return 1.0
    if raw_score >= 0.2:
        return 0.5
    return 0.0


def score_persona_reply(
    scenario: Dict[str, Any],
    prior_user_turns: List[str],
    prior_assistant_turns: List[str],
    reply: str,
) -> float:
    source_texts = [
        scenario.get("character", ""),
        " ".join(scenario.get("hard_rules", [])),
        *prior_user_turns,
        *prior_assistant_turns,
    ]
    anchor_terms = extract_salient_terms(source_texts, top_n=8)
    raw_score = overlap_score(reply, anchor_terms)

    if raw_score >= 0.5:
        return 1.0
    if raw_score >= 0.2:
        return 0.5
    return 0.0

def load_scenario_map() -> Dict[str, Any]:
    if not SCENARIOS_FILE.exists():
        return {}
    with open(SCENARIOS_FILE, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    return {normalize_scenario_key(s["title"]): s for s in scenarios}

def score_response(
    scenario: Dict[str, Any],
    user_input: str,
    prior_user_turns: List[str],
    prior_assistant_turns: List[str],
    reply: str,
) -> Dict[str, float]:
    """Applies type-specific scoring logic to a single response."""
    reply_lower = reply.lower()
    scores = {
        "memory_score": 0.0,
        "persona_score": 0.0,
        "constraint_score": 1.0,  # Start at 1.0, penalize on failure
        "ai_penalty": 0.0
    }

    stype = scenario.get("type")

    # 1. Memory Scoring
    if stype == "memory":
        scores["memory_score"] = score_memory_reply(
            user_input,
            prior_user_turns,
            reply,
        )

    # 1b. Persona/role consistency scoring
    scores["persona_score"] = score_persona_reply(
        scenario,
        prior_user_turns,
        prior_assistant_turns,
        reply,
    )

    # 2. Constraint/Jailbreak Scoring
    # NOTE: This is a coarse heuristic. One forbidden keyword nukes the score.
    # Future iterations should consider frequency or context-aware checking.
    if stype == "constraint" and "forbidden_keywords" in scenario:
        for forbidden in scenario["forbidden_keywords"]:
            if forbidden.lower() in reply_lower:
                scores["constraint_score"] = 0.0
                break

    # 3. AI Identity Handling
    # Most scenarios penalize "I am an AI" mentions. 'ai_identity' skips this.
    if stype != "ai_identity" and any(p in reply_lower for p in BANNED_AI_PHRASES):
        scores["ai_penalty"] = 1.0

    return scores


def compute_final_score(
    scenario: Dict[str, Any],
    user_input: str,
    prior_user_turns: List[str],
    s_scores: Dict[str, float],
    base_pass: float,
) -> float:
    scenario_type = scenario.get("type")

    if scenario_type == "memory":
        if is_memory_probe(user_input, prior_user_turns):
            return (
                (0.6 * s_scores["memory_score"])
                + (0.2 * s_scores["persona_score"])
                + (0.2 * base_pass)
                - (s_scores["ai_penalty"] * 0.5)
            )
        return (
            (0.5 * base_pass)
            + (0.5 * s_scores["persona_score"])
            - (s_scores["ai_penalty"] * 0.5)
        )

    if scenario_type == "constraint":
        return (
            (0.5 * base_pass) + (0.5 * s_scores["persona_score"])
        ) * s_scores["constraint_score"] - (s_scores["ai_penalty"] * 0.8)

    if scenario_type == "ai_identity":
        return (0.4 * base_pass) + (0.6 * s_scores["persona_score"])

    return (
        (0.5 * base_pass) + (0.5 * s_scores["persona_score"])
        - (s_scores["ai_penalty"] * 0.8)
    )


def score_csv_row(
    row: Dict[str, str],
    scenario_map: Dict[str, Any],
    prior_user_turns: Dict[tuple[str, str], List[str]],
    prior_assistant_turns: Dict[tuple[str, str], List[str]],
) -> Dict[str, Any] | None:
    scenario_key = normalize_scenario_key(row["scenario_title"])
    scenario = scenario_map.get(scenario_key)
    if scenario is None:
        return None

    key = (row["model_file"], scenario_key)
    previous_users = prior_user_turns.get(key, [])
    previous_assistant = prior_assistant_turns.get(key, [])
    s_scores = score_response(
        scenario,
        row["user_input"],
        previous_users,
        previous_assistant,
        row["raw_output"],
    )
    base_pass = 1.0 if row["passed_basic_checks"] == "True" else 0.0
    final = compute_final_score(scenario, row["user_input"], previous_users, s_scores, base_pass)

    row.update(s_scores)
    row["final_score"] = max(0.0, min(1.0, final))
    prior_user_turns[key] = previous_users + [row["user_input"]]
    prior_assistant_turns[key] = previous_assistant + [row["raw_output"]]
    return row

def calculate_scores():
    if not RESULTS_FILE.exists():
        print("No results file found. Run benchmark_runner first.")
        return

    scenario_map = load_scenario_map()
    
    # We'll read the CSV and write a new one with scoring columns
    rows = []
    skipped_rows = 0
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        base_fieldnames = reader.fieldnames or []
        fieldnames = base_fieldnames + ["memory_score", "persona_score", "constraint_score", "ai_penalty", "final_score"]
        
        # Track prior user turns per (model, scenario).
        prior_user_turns = {}
        prior_assistant_turns = {}

        for row in reader:
            scored_row = score_csv_row(row, scenario_map, prior_user_turns, prior_assistant_turns)
            if scored_row is None:
                skipped_rows += 1
                continue
            rows.append(scored_row)

    # Write scored results
    PROJECT_ROOT.joinpath("results").mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Scored results written to: {OUTPUT_FILE}")
    if skipped_rows:
        print(f"Skipped {skipped_rows} row(s) with scenario titles not found in current scenarios.json")

    # Summary reporting
    import pandas as pd
    df = pd.DataFrame(rows)
    df["final_score"] = pd.to_numeric(df["final_score"])
    
    summary = df.groupby("model_file")["final_score"].mean().sort_values(ascending=False)
    print("\n--- Final Model Rankings (Scored) ---")
    print(summary)

if __name__ == "__main__":
    calculate_scores()
