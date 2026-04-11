from __future__ import annotations

# checks rule exists
def _rule_exists(scenario: dict, phrase: str) -> bool:
    phrase = phrase.lower()
    return any(phrase in rule.lower() for rule in scenario.get("hard_rules", []))


def _mentions_ai(reply: str) -> bool:
    lowered = reply.lower()
    # this is just bank of words, maybe better technique needed
    banned_phrases = [
        "as an ai",
        "i am an ai",
        "i'm an ai",
        "language model",
        "ai assistant",
        "artificial intelligence",
    ]
    return any(phrase in lowered for phrase in banned_phrases)


def evaluate_reply(scenario: dict, reply: str | None) -> dict:
    reply_text = (reply or "").strip()
    word_count = len(reply_text.split())

    checks = {
        "non_empty": bool(reply_text),
        "mentions_ai": _mentions_ai(reply_text),
        "under_30_words": True,
    }

    if _rule_exists(scenario, "under 30 words"):
        checks["under_30_words"] = word_count <= 30

    failure_reasons: list[str] = []

    if not checks["non_empty"]:
        failure_reasons.append("empty_reply")

    if _rule_exists(scenario, "do not mention being an ai") and checks["mentions_ai"]:
        failure_reasons.append("mentioned_ai")

    if not checks["under_30_words"]:
        failure_reasons.append("too_many_words")

    passed_basic_checks = len(failure_reasons) == 0

    return {
        "passed_basic_checks": passed_basic_checks,
        "failure_reason": "; ".join(failure_reasons) if failure_reasons else "",
        "word_count": word_count,
        **checks,
    }
