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
    scenario_texts = [scenario.get("character", ""), *scenario.get("hard_rules", [])]
    order_state = build_state([], prior_user_turns)
    world_state = build_state(scenario_texts, prior_user_turns)
    fact_parts: List[str] = []
    asks_total_only = "total items" in lowered or ("how many" in lowered and "last time" not in lowered)

    # use exact fact snippets first when the question clearly points at them
    if "my name" in lowered:
        user_names = extract_names(" ".join(prior_user_turns))
        if user_names:
            fact_parts.append(f"Your name is {user_names[-1]}")

    if "your name" in lowered or ("name again" in lowered and "my name" not in lowered):
        assistant_names = extract_names(" ".join(scenario_texts))
        if assistant_names:
            fact_parts.append(f"My name is {assistant_names[0]}")

    if "balance" in lowered and "balance" in world_state:
        fact_parts.append(f"Your balance is ${world_state['balance']}")

    if asks_total_only:
        total = sum(amount for key, amount in order_state.items() if key != "balance" and amount > 0)
        if total:
            fact_parts.append(f"You ordered {total} items in total")

    if not asks_total_only and any(cue in lowered for cue in ("tab", "order", "final order", "bought", "last time")):
        order_keys = [key for key in order_state if key != "balance" and order_state[key] >= 0]
        if order_keys:
            summary = format_counted_items(order_state, sorted(order_keys))
            if summary:
                if "last time" in lowered:
                    fact_parts.append(f"Last time you bought {summary}")
                else:
                    fact_parts.append(f"Your order is {summary}")

    if any(cue in lowered for cue in ("stock", "current number", "in stock")):
        stock_keys = [key for key in world_state if key != "balance" and world_state[key] >= 0]
        if stock_keys:
            summary = format_counted_items(world_state, sorted(stock_keys))
            if summary:
                fact_parts.append(f"The current stock is {summary}")

    references: List[str] = []
    if fact_parts:
        references.append(". ".join(fact_parts) + ".")

    # fall back to relevant snippets when the turn is about broader context not exact state
    if any(cue in lowered for cue in ("schedule", "calendar", "event", "quest", "spell", "dragon", "close today")):
        references.extend(select_relevant_context(user_input, [*scenario_texts, *prior_user_turns], similarity, limit=2))

    if not references:
        references.extend(select_relevant_context(user_input, [*scenario_texts, *prior_user_turns], similarity, limit=2))

    return unique_in_order(references)
