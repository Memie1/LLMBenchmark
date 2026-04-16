from __future__ import annotations

import re
from typing import Any, Callable, Dict, Iterable, List

# marks actions written by the user that the ai should react to
ADD_ACTIONS = {"received", "receive", "added", "add", "put", "ordered", "purchased", 
               "bought", "deposit", "deposited", "found"}
# marks where the user is losing items or money so the ai can react with empathy and avoid bad math
SUBTRACT_ACTIONS = {"sold", "remove", "removed", "spent", "lost"}
# Written in hard rules there are usually acknowledgement of how much the ai has of an item in stock
SET_CUES = ("start off with", "starts off with", "currently have", "balance of", 
            "stock is", "schedule is empty")
# When the user changes their mind
REPLACE_CUES = ("make that", "set the", "reduce", "change to")
# remove filler words 
IGNORE_ITEM_WORDS = {"the", "my", "your", "our", "their", "fresh", "new", "just", 
                     "please", "confirm", "well", "back"}
# these small vocab lists keep the state parser deterministic without pulling in a full judge model
WORD_NUMBERS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
MY_NAME_CUE = "my name"
YOUR_NAME_CUE = "your name"
LAST_TIME_CUE = "last time"
ORDER_QUERY_CUES = ("tab", "order", "final order", "bought", LAST_TIME_CUE)
STOCK_QUERY_CUES = ("stock", "current number", "in stock")
BROAD_CONTEXT_CUES = ("schedule", "calendar", "event", "quest", "spell", "dragon", "close today")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

# create list of seen values and ordered values in one pass while preserving order and skipping empties
def unique_in_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered

def tokenize(text: str) -> List[str]:
    # keep tokenization simple because this is only used for lightweight state parsing
    return re.findall(r"[a-z0-9'$:-]+", text.lower())


def parse_number(token: str) -> int | None:
    # support both digits and small number words because scenarios use both styles
    cleaned = token.lower().lstrip("$")
    if cleaned in WORD_NUMBERS:
        return WORD_NUMBERS[cleaned]
    if cleaned.isdigit():
        return int(cleaned)
    return None


def singularize(word: str) -> str:
    # collapse common plurals so state keys stay stable across turns
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ves") and len(word) > 4:
        return word[:-3] + "f"
    if word.endswith("es") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and len(word) > 3 and not word.endswith("ss"):
        return word[:-1]
    return word


def pluralize(item: str, count: int) -> str:
    # rebuild readable item names when we turn state back into reference text
    if count == 1:
        return item
    if item.endswith("s"):
        return item
    if item.endswith("y") and len(item) > 2:
        return item[:-1] + "ies"
    return item + "s"


def extract_names(text: str) -> List[str]:
    # names are treated as exact facts so they get explicit extraction instead of semantic guessing
    names = [match.group(1).strip() for match in re.finditer(
        r"\b(?:I am|I'm|my name is|name is|named)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text,
    )]
    return unique_in_order(names)


def extract_counted_items(text: str) -> List[Dict[str, Any]]:
    # pull out simple number + item pairs so state updates can stay deterministic
    raw_tokens = tokenize(text)
    items: List[Dict[str, Any]] = []

    for index, token in enumerate(raw_tokens):
        amount = parse_number(token)
        if amount is None or token.endswith("am") or token.endswith("pm"):
            continue

        window_before = raw_tokens[max(0, index - 3):index]
        window_after = raw_tokens[index + 1:index + 4]
        # balances need their own key because money updates should not look like normal inventory items
        if "balance" in window_before or "account" in window_before or "balance" in window_after or "account" in window_after:
            items.append({"item": "balance", "amount": amount, "index": index})
            continue

        item_tokens: List[str] = []
        next_index = index + 1
        while next_index < len(raw_tokens) and len(item_tokens) < 3:
            next_token = raw_tokens[next_index]
            if parse_number(next_token) is not None or next_token in {"and", "or", "but"}:
                break
            if next_token in IGNORE_ITEM_WORDS:
                next_index += 1
                continue
            item_tokens.append(singularize(next_token))
            next_index += 1

        if not item_tokens:
            continue

        item_name = "bread" if "bread" in item_tokens else item_tokens[0]
        items.append({"item": item_name, "amount": amount, "index": index})

    return items


def apply_turn_update(text: str, state: Dict[str, int]) -> None:
    # apply one turn at a time so later corrections overwrite earlier state cleanly
    lowered = normalize_text(text)
    tokens = tokenize(text)
    counted_items = extract_counted_items(text)

    for match in re.finditer(r"\bremove\s+(?:the\s+)?([a-z]+)\b", lowered):
        state[singularize(match.group(1))] = 0

    # explicit replacements win before we inspect looser add/subtract language
    for match in re.finditer(r"\b(?:reduce|set|change)\s+(?:the\s+)?([a-z]+)\s+to\s+([a-z0-9$]+)\b", lowered):
        amount = parse_number(match.group(2))
        if amount is not None:
            state[singularize(match.group(1))] = amount

    mode = "set" if any(cue in lowered for cue in SET_CUES) else None
    if any(cue in lowered for cue in REPLACE_CUES):
        mode = "replace"

    # local verb windows keep mixed turns like "sold X and received Y" workable
    for counted_item in counted_items:
        item = counted_item["item"]
        amount = counted_item["amount"]
        window = tokens[max(0, counted_item["index"] - 4):counted_item["index"]]

        if item == "balance" and "deposited" in window:
            state[item] = state.get(item, 0) + amount
            continue

        if any(token in SUBTRACT_ACTIONS for token in window):
            state[item] = state.get(item, 0) - amount
            continue

        if any(token in ADD_ACTIONS for token in window):
            state[item] = state.get(item, 0) + amount
            continue

        if mode in {"set", "replace"}:
            state[item] = amount


def build_state(base_texts: Iterable[str], update_texts: Iterable[str]) -> Dict[str, int]:
    # start from scenario facts first then replay the user updates in order
    state: Dict[str, int] = {}

    for text in base_texts:
        apply_turn_update(text, state)

    for text in update_texts:
        apply_turn_update(text, state)

    return state


def format_counted_items(state: Dict[str, int], keys: Iterable[str]) -> str:
    # turn compact state back into a short phrase the semantic scorer can compare against
    parts: List[str] = []
    for key in keys:
        amount = state.get(key)
        if amount is None or amount < 0:
            continue
        parts.append(f"{amount} {pluralize(key, amount)}")
    return " and ".join(parts)


def extract_total_items(text: str) -> int | None:
    # totals are checked separately because replies often answer with just one number
    tokens = tokenize(text)
    for index, token in enumerate(tokens):
        if token != "items" or index == 0:
            continue
        amount = parse_number(tokens[index - 1])
        if amount is not None:
            return amount
    return None


def extract_item_amount_sets(text: str) -> Dict[str, set[int]]:
    # track all claimed amounts per item so contradictory replies can fail exact state checks
    amounts: Dict[str, set[int]] = {}
    for counted_item in extract_counted_items(text):
        amounts.setdefault(counted_item["item"], set()).add(counted_item["amount"])
    return amounts


def build_memory_state_context(
    scenario: Dict[str, Any],
    prior_user_turns: List[str],
) -> tuple[List[str], Dict[str, int], Dict[str, int]]:
    scenario_texts = [scenario.get("character", ""), *scenario.get("hard_rules", [])]
    order_state = build_state([], prior_user_turns)
    world_state = build_state(scenario_texts, prior_user_turns)
    return scenario_texts, order_state, world_state


def is_total_only_query(lowered: str) -> bool:
    return "total items" in lowered or ("how many" in lowered and LAST_TIME_CUE not in lowered)


def is_order_query(lowered: str, asks_total_only: bool) -> bool:
    return not asks_total_only and any(cue in lowered for cue in ORDER_QUERY_CUES)


def is_stock_query(lowered: str) -> bool:
    return any(cue in lowered for cue in STOCK_QUERY_CUES)


def collect_name_checks(
    lowered: str,
    scenario_texts: List[str],
    prior_user_turns: List[str],
) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []

    if MY_NAME_CUE in lowered:
        user_names = extract_names(" ".join(prior_user_turns))
        if user_names:
            checks.append({"kind": "name", "key": "user_name", "expected": user_names[-1]})

    if YOUR_NAME_CUE in lowered or ("name again" in lowered and MY_NAME_CUE not in lowered):
        assistant_names = extract_names(" ".join(scenario_texts))
        if assistant_names:
            checks.append({"kind": "name", "key": "assistant_name", "expected": assistant_names[0]})

    return checks


def collect_total_check(order_state: Dict[str, int]) -> Dict[str, Any]:
    total = sum(amount for key, amount in order_state.items() if key != "balance" and amount > 0)
    return {"kind": "total_items", "key": "total_items", "expected": total}


def collect_item_checks(state: Dict[str, int], domain: str) -> List[Dict[str, Any]]:
    keys = [key for key in state if key != "balance" and state[key] >= 0]
    return [
        {"kind": "item", "domain": domain, "key": key, "expected": state[key]}
        for key in sorted(keys)
    ]


def collect_name_fact_parts(
    lowered: str,
    scenario_texts: List[str],
    prior_user_turns: List[str],
) -> List[str]:
    parts: List[str] = []

    if MY_NAME_CUE in lowered:
        user_names = extract_names(" ".join(prior_user_turns))
        if user_names:
            parts.append(f"Your name is {user_names[-1]}")

    if YOUR_NAME_CUE in lowered or ("name again" in lowered and MY_NAME_CUE not in lowered):
        assistant_names = extract_names(" ".join(scenario_texts))
        if assistant_names:
            parts.append(f"My name is {assistant_names[0]}")

    return parts


def collect_balance_fact_part(lowered: str, world_state: Dict[str, int]) -> str | None:
    if "balance" in lowered and "balance" in world_state:
        return f"Your balance is ${world_state['balance']}"
    return None


def collect_total_fact_part(order_state: Dict[str, int]) -> str | None:
    total = sum(amount for key, amount in order_state.items() if key != "balance" and amount > 0)
    return f"You ordered {total} items in total"


def collect_order_fact_part(lowered: str, order_state: Dict[str, int]) -> str | None:
    summary = format_counted_items(order_state, sorted(key for key in order_state if key != "balance" and order_state[key] >= 0))
    if not summary:
        return None
    if LAST_TIME_CUE in lowered:
        return f"Last time you bought {summary}"
    return f"Your order is {summary}"


def collect_stock_fact_part(world_state: Dict[str, int]) -> str | None:
    summary = format_counted_items(world_state, sorted(key for key in world_state if key != "balance" and world_state[key] >= 0))
    if not summary:
        return None
    return f"The current stock is {summary}"


def score_single_memory_check(check: Dict[str, Any], reply_names: List[str], reply_item_amounts: Dict[str, set[int]], reply_total: int | None) -> bool:
    if check["kind"] == "name":
        return normalize_text(str(check["expected"])) in reply_names
    if check["kind"] == "total_items":
        return reply_total == check["expected"]
    amounts = reply_item_amounts.get(check["key"], set())
    return amounts == {check["expected"]}


def structured_reference_score(reference: str, reply: str) -> float:
    # exact facts still get a deterministic boost even when the main scorer is semantic
    expected_checks = 0
    matched_checks = 0
    lowered_reply = normalize_text(reply)

    for name in extract_names(reference):
        expected_checks += 1
        if normalize_text(name) in lowered_reply:
            matched_checks += 1

    expected_items = {item["item"]: item["amount"] for item in extract_counted_items(reference)}
    reply_items = {item["item"]: item["amount"] for item in extract_counted_items(reply)}
    # exact item counts get their own pass so good factual replies are not dragged down by paraphrasing
    for item, amount in expected_items.items():
        expected_checks += 1
        if reply_items.get(item) == amount:
            matched_checks += 1

    reference_total = extract_total_items(reference)
    if reference_total is not None:
        expected_checks += 1
        if extract_total_items(reply) == reference_total:
            matched_checks += 1

    if expected_checks == 0:
        return 0.0

    return matched_checks / expected_checks


def build_memory_state_checks(
    scenario: Dict[str, Any],
    user_input: str,
    prior_user_turns: List[str],
) -> List[Dict[str, Any]]:
    # build the exact state facts that the current memory question is asking the model to preserve
    lowered = normalize_text(user_input)
    scenario_texts, order_state, world_state = build_memory_state_context(scenario, prior_user_turns)
    asks_total_only = is_total_only_query(lowered)
    checks = collect_name_checks(lowered, scenario_texts, prior_user_turns)

    if "balance" in lowered and "balance" in world_state:
        checks.append({"kind": "item", "domain": "world", "key": "balance", "expected": world_state["balance"]})

    if asks_total_only:
        checks.append(collect_total_check(order_state))

    if is_order_query(lowered, asks_total_only):
        checks.extend(collect_item_checks(order_state, "order"))

    if is_stock_query(lowered):
        checks.extend(collect_item_checks(world_state, "world"))

    return checks


def score_memory_state_diff(checks: List[Dict[str, Any]], reply: str) -> float:
    # score whether the reply preserves the exact state facts requested by the memory question
    if not checks:
        return 0.0

    reply_names = [normalize_text(name) for name in extract_names(reply)]
    reply_item_amounts = extract_item_amount_sets(reply)
    reply_total = extract_total_items(reply)
    matched_checks = sum(
        1
        for check in checks
        if score_single_memory_check(check, reply_names, reply_item_amounts, reply_total)
    )

    return matched_checks / len(checks)


def select_relevant_context(
    query: str,
    texts: Iterable[str],
    similarity: Callable[[str, str], float],
    limit: int = 2,
) -> List[str]:
    # keep only the most relevant snippets so the reference answer stays short
    ranked = sorted(
        ((similarity(query, text), text.strip()) for text in texts if text.strip()),
        key=lambda item: item[0],
        reverse=True,
    )
    return [text for score, text in ranked[:limit] if score > 0.1]


def build_memory_references(
    scenario: Dict[str, Any],
    user_input: str,
    prior_user_turns: List[str],
    similarity: Callable[[str, str], float], ) -> List[str]:
    # build a short expected answer from exact state where possible and semantic context otherwise
    lowered = normalize_text(user_input)
    scenario_texts, order_state, world_state = build_memory_state_context(scenario, prior_user_turns)
    asks_total_only = is_total_only_query(lowered)
    fact_parts = collect_name_fact_parts(lowered, scenario_texts, prior_user_turns)

    balance_fact = collect_balance_fact_part(lowered, world_state)
    if balance_fact:
        fact_parts.append(balance_fact)

    if asks_total_only:
        fact_parts.append(collect_total_fact_part(order_state))

    if is_order_query(lowered, asks_total_only):
        order_fact = collect_order_fact_part(lowered, order_state)
        if order_fact:
            fact_parts.append(order_fact)

    if is_stock_query(lowered):
        stock_fact = collect_stock_fact_part(world_state)
        if stock_fact:
            fact_parts.append(stock_fact)

    references: List[str] = []
    if fact_parts:
        references.append(". ".join(fact_parts) + ".")

    # fall back to relevant snippets when the turn is about broader context not exact state
    if any(cue in lowered for cue in BROAD_CONTEXT_CUES):
        references.extend(select_relevant_context(user_input, [*scenario_texts, *prior_user_turns], similarity, limit=2))

    if not references:
        references.extend(select_relevant_context(user_input, [*scenario_texts, *prior_user_turns], similarity, limit=2))

    return unique_in_order(references)
