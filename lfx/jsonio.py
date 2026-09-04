"""Pull a JSON object out of model output."""

import json
import re


def extract_json(text):
    """First balanced {...} in the text, parsed. None if there isn't one.

    Base models wrap their answer in prose or code fences, and a naive
    `text[text.find("{"):text.rfind("}")]` breaks on braces inside strings.
    """
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth, instr, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if instr:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                instr = False
            continue
        if ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None
