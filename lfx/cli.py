"""Extract the logical structure of an argument. Runs locally on the Mac via MLX.

    logicalform "All men are mortal. Socrates is a man. So Socrates is mortal."
    pbpaste | logicalform
    logicalform --repl            # paste passages one after another while reading
    logicalform --json < essay.txt

The system prompt is imported from format_for_training.py rather than restated, so
it cannot drift from the prompt the model was fine-tuned on — a mismatch there
degrades output quietly, with no error to notice.
"""

import argparse
import json
import sys

from lfx.jsonio import extract_json
from lfx.prompts import SYSTEM
from lfx.schema import FAMILY, inconsistency

# 8-bit over 4-bit: same speed class (~0.6s/passage), perfect schema validity on
# test_real, better suppressed-premise accuracy. 4-bit scored 1 example lower on
# form accuracy out of 34 — not resolvable at that sample size.
DEFAULT_MODEL = "models/mlx_v2_8bit"

def looks_like_argument(text):
    """Reject input the model would only hallucinate over.

    The corpus contains 840 arguments and zero non-arguments, so the model has
    never seen a passage with nothing to extract. Given one it fabricates a
    plausible argument from pretraining rather than failing — "r" produces a
    textbook syllogism about club membership. Cheaper to refuse here than to
    retrain with negative examples.
    """
    words = text.split()
    if len(words) < 6:
        return "too short to contain an argument"
    if len(set(w.lower().strip(".,;:!?") for w in words)) < 4:
        return "too few distinct words"
    return None


def render(d, colour=True):
    C = (lambda s, c: f"\033[{c}m{s}\033[0m") if colour else (lambda s, c: s)
    out = []
    for i, p in enumerate(d.get("premises", []), 1):
        out.append(f"  {C(f'P{i}', '2;37')}  {p}")
    out.append(f"  {C('∴ ', '1;36')} {C(d.get('conclusion', ''), '1')}")
    out.append("")
    form = d.get("form", "?")
    fam = FAMILY.get(form, "")
    tag = C(form, "1;31" if "FALLACY" in fam.upper() else "1;36")
    out.append(f"  {d.get('argument_type', '?')} · {tag}" + (f"  {C(f'({fam})', '2;37')}" if fam else ""))
    if d.get("suppressed_premise"):
        out.append(f"  {C('unstated:', '2;37')} {d['suppressed_premise']}")
    if (bad := inconsistency(d)):
        out.append("")
        out.append(f"  {C('inconsistent:', '1;33')} {bad}")
        out.append(f"  {C('the form label here is unreliable — read the argument yourself', '2;37')}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Extract argument structure. Runs locally.")
    ap.add_argument("text", nargs="*", help="the passage; omit to read stdin")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--json", action="store_true", help="raw JSON, for piping")
    ap.add_argument("--repl", action="store_true", help="keep the model loaded, read passages in a loop")
    ap.add_argument("--max-tokens", type=int, default=768)
    args = ap.parse_args()

    from mlx_lm import generate, load                 # imported late: ~1s, skip it for --help
    model, tok = load(args.model)
    sys_msg = SYSTEM

    def run(passage):
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user", "content": passage}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        raw = generate(model, tok, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        parsed = extract_json(raw)
        if args.json:
            if parsed and (bad := inconsistency(parsed)):
                parsed = {**parsed, "_inconsistent": bad}
            print(json.dumps(parsed if parsed else {"error": "unparseable", "raw": raw},
                             ensure_ascii=False))
        elif parsed:
            print(render(parsed, colour=sys.stdout.isatty()))
        else:
            print("could not parse a JSON object from the model:\n" + raw, file=sys.stderr)

    if args.repl:
        print(f"loaded {args.model} — paste a passage, blank line to submit, Ctrl-D to quit\n")
        while True:
            lines = []
            try:
                while (line := input("› " if not lines else "  ")) != "":
                    lines.append(line)
            except EOFError:
                print()
                return
            if lines:
                print()
                run(" ".join(lines))
                print()
        return

    passage = (" ".join(args.text) if args.text else sys.stdin.read()).strip()
    if not passage:
        ap.error("no passage given (pass text, or pipe it on stdin)")
    if (why := looks_like_argument(passage)):
        print(f"refusing: {why}. The model has only ever seen real arguments and "
              f"will invent one rather than report that there is none.", file=sys.stderr)
        sys.exit(2)
    run(passage)


if __name__ == "__main__":
    main()
