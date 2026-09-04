"""Auto-label a raw argument corpus with Gemini via Vertex AI.

Reads a corpus of passages, asks Gemini for a schema-conforming label for each,
and writes candidate labels alongside the raw text for a human review pass.
The schema, the labeler instruction and the retry logic all live in lfx.vertex.

    python pipeline/label/label_corpus.py --corpus data/raw/seed_corpus.jsonl \
        --out data/interim/labeled_candidates.jsonl
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from lfx.vertex import build_examples, client, label_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/raw/seed_corpus.jsonl")
    ap.add_argument("--fewshot", default="data/raw/fewshot.jsonl")
    ap.add_argument("--out", default="data/interim/labeled_candidates.jsonl")
    ap.add_argument("--limit", type=int, help="label at most N passages")
    ap.add_argument("--force", action="store_true", help="ignore existing output, relabel all")
    ap.add_argument("--workers", type=int, default=6, help="parallel requests")
    args = ap.parse_args()

    rows = [json.loads(line) for line in open(args.corpus)]

    done = set()
    if os.path.exists(args.out) and not args.force:
        with open(args.out) as fh:
            done = {json.loads(line)["id"] for line in fh}
    todo = [r for r in rows if r["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]

    print(f"{len(rows)} passages, {len(done)} already labeled, {len(todo)} to do")
    if not todo:
        return

    cl = client()
    examples = build_examples(args.fewshot)

    mode = "w" if args.force else "a"
    out = open(args.out, mode)
    lock = threading.Lock()
    n = [0]

    def one(row):
        try:
            label = label_one(cl, examples, row["text"])
        except Exception as exc:  # noqa: BLE001 - record the miss, keep the batch going
            print(f"  ! {row['id']}: {type(exc).__name__} {exc}"[:160], file=sys.stderr)
            return
        rec = {k: row[k] for k in ("id", "category", "source", "text") if k in row}
        rec.update({"label": label, "reviewed": False, "review_notes": ""})
        for k in ("intended_form", "domain", "topic"):   # carried through for triage
            if k in row:
                rec[k] = row[k]
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            n[0] += 1
            if n[0] % 25 == 0 or n[0] == len(todo):
                print(f"  {n[0]}/{len(todo)}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, todo))
    out.close()
    print(f"\nwrote {n[0]} -> {args.out}")


if __name__ == "__main__":
    main()
