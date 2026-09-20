"""Derive an argument's form from its logical skeleton, by rule rather than judgement.

The model this project trains is asked to look at prose and name which of 22 forms
it instantiates. That is the thing language models are worst at, and the error
pattern shows it: `modus ponens` and `affirming the consequent` differ by which
term is affirmed, and a model that reads the passage correctly still confuses them
because it is pattern-matching on vibes rather than on structure.

But for roughly half the enum the form is FULLY DETERMINED by the skeleton. Given

    premises   P -> Q,  Q
    conclusion P

there is nothing to judge. It is affirming the consequent, necessarily, and code
can say so with certainty. So the model's job becomes translation -- prose to
skeleton -- and this module does the reasoning. Translation is what language
models are good at.

Matching is STRUCTURAL, not literal. `_X` and `_Y` are metavariables that bind to
any subformula, so

    "if it rains and the pitch floods, the match is off"; "it rains and the pitch
    floods"; therefore "the match is off"

is modus ponens with _X bound to a conjunction. A checker that only handled atoms
would miss most real arguments.

Two skeleton languages, because one does not cover the ground:

  propositional   ~ | & ->, for the conditional and disjunctive forms
  categorical     All/No/Some A are B, for syllogisms, where propositional logic
                  cannot see the quantifier structure that makes them valid

Both use CANONICAL LABELS -- P, Q for propositions, S, M, P for terms -- with the
prose kept in a separate mapping. That is not decoration. A first attempt matched
terms on their surface text and could not see that "men" and "a man" are the same
term, so Barbara failed on Socrates. Fuzzy matching would have papered over it and
introduced false positives in exchange; a checker whose whole value is certainty
cannot guess. Labels make term identity exact, and inconsistent labelling is
reported as no-match rather than silently repaired.

Everything here is deterministic and total: it returns a form or it returns None.
None means "this skeleton matches no known form", which is information -- it is
how a malformed translation announces itself.

    python -m lfx.formal        # runs the self-tests
"""

import itertools

# ---------------------------------------------------------------- propositional

BINARY = {"->", "|", "&"}


def tokenize(s):
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif s.startswith("->", i):
            out.append("->")
            i += 2
        elif c in "~|&()":
            out.append(c)
            i += 1
        elif c.isalnum() or c == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            out.append(s[i:j])
            i = j
        else:
            raise ValueError(f"unexpected character {c!r} in {s!r}")
    return out


class _Parser:
    """Recursive descent. Precedence, loosest first: -> then | then & then ~."""

    def __init__(self, toks, src):
        self.t, self.i, self.src = toks, 0, src

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else None

    def eat(self, expect=None):
        tok = self.peek()
        if tok is None or (expect is not None and tok != expect):
            raise ValueError(f"expected {expect or 'token'}, got {tok!r} in {self.src!r}")
        self.i += 1
        return tok

    def formula(self):
        left = self.disjunction()
        if self.peek() == "->":
            self.eat("->")
            return ("->", left, self.formula())  # right associative
        return left

    def disjunction(self):
        left = self.conjunction()
        while self.peek() == "|":
            self.eat("|")
            left = ("|", left, self.conjunction())
        return left

    def conjunction(self):
        left = self.unary()
        while self.peek() == "&":
            self.eat("&")
            left = ("&", left, self.unary())
        return left

    def unary(self):
        if self.peek() == "~":
            self.eat("~")
            return ("~", self.unary())
        return self.primary()

    def primary(self):
        tok = self.peek()
        if tok == "(":
            self.eat("(")
            f = self.formula()
            self.eat(")")
            return f
        if tok is None or tok in BINARY or tok in "~|&()":
            raise ValueError(f"expected an atom, got {tok!r} in {self.src!r}")
        return ("atom", self.eat())


def parse(s):
    """Parse a propositional formula. Raises ValueError on anything malformed."""
    p = _Parser(tokenize(s), s)
    f = p.formula()
    if p.peek() is not None:
        raise ValueError(f"trailing {p.peek()!r} in {s!r}")
    return normalize(f)


def normalize(f):
    """Collapse double negation, so ~~P and P are the same skeleton."""
    if f[0] == "~":
        inner = normalize(f[1])
        return inner[1] if inner[0] == "~" else ("~", inner)
    if f[0] in BINARY:
        return (f[0], normalize(f[1]), normalize(f[2]))
    return f


def is_meta(f):
    return f[0] == "atom" and f[1].startswith("_")


def match(pattern, formula, bound):
    """Unify `pattern` against `formula`. Returns bindings, or None if no match."""
    if is_meta(pattern):
        name = pattern[1]
        if name in bound:
            return bound if bound[name] == formula else None
        out = dict(bound)
        out[name] = formula
        return out
    if pattern[0] != formula[0]:
        return None
    if pattern[0] == "atom":
        return bound if pattern[1] == formula[1] else None
    if pattern[0] == "~":
        return match(pattern[1], formula[1], bound)
    left = match(pattern[1], formula[1], bound)
    return None if left is None else match(pattern[2], formula[2], left)


def match_argument(pat_premises, pat_conclusion, premises, conclusion):
    """Premises are a SET: try every ordering before concluding there is no match."""
    if len(pat_premises) != len(premises):
        return None
    for order in itertools.permutations(premises):
        bound, ok = {}, True
        for pat, prem in zip(pat_premises, order):
            bound = match(pat, prem, bound)
            if bound is None:
                ok = False
                break
        if ok:
            final = match(pat_conclusion, conclusion, bound)
            if final is not None:
                return final
    return None


# Each entry is (form, premise patterns, conclusion pattern). Both valid forms and
# formal fallacies are here: the point is to name the shape, not to endorse it.
PATTERNS = [
    ("modus ponens",             ["_X -> _Y", "_X"],        "_Y"),
    ("modus tollens",            ["_X -> _Y", "~_Y"],       "~_X"),
    ("affirming the consequent", ["_X -> _Y", "_Y"],        "_X"),
    ("denying the antecedent",   ["_X -> _Y", "~_X"],       "~_Y"),
    ("hypothetical syllogism",   ["_X -> _Y", "_Y -> _Z"],  "_X -> _Z"),
    ("disjunctive syllogism",    ["_X | _Y", "~_X"],        "_Y"),
    ("disjunctive syllogism",    ["_X | _Y", "~_Y"],        "_X"),
    ("reductio ad absurdum",     ["_X -> (_Y & ~_Y)"],      "~_X"),
]

_COMPILED = [(name, [parse(p) for p in prems], parse(concl))
             for name, prems, concl in PATTERNS]


def classify_propositional(premises, conclusion):
    """Every form this skeleton matches. Usually one; [] means none.

    Returning a list rather than a single answer is deliberate -- if two forms ever
    match the same skeleton that is a bug in the pattern set, and silently taking
    the first would hide it.
    """
    prems = [parse(p) for p in premises]
    concl = parse(conclusion)

    # Checked first: a conclusion that merely restates a premise is circular
    # whatever else its shape suggests.
    if any(p == concl for p in prems):
        return ["begging the question"]

    found = []
    for name, pat_prems, pat_concl in _COMPILED:
        if match_argument(pat_prems, pat_concl, prems, concl) is not None:
            if name not in found:
                found.append(name)
    return found


# ----------------------------------------------------------------- categorical

QUANTIFIERS = {"all": "A", "no": "E", "some": "I", "some not": "O"}

_ARTICLES = ("a ", "an ", "the ")


def _term(s):
    """Canonicalise a term label. Articles only -- no stemming, no synonyms."""
    t = " ".join(s.strip().split()).lower()
    for art in _ARTICLES:
        if t.startswith(art):
            return t[len(art):].strip()
    return t

# The 15 moods valid without assuming the subject class is non-empty. The other 9
# of the traditional 24 need existential import, which is a metaphysical
# commitment this project has no reason to take on.
VALID_MOODS = {
    1: {"AAA", "EAE", "AII", "EIO"},
    2: {"AEE", "EAE", "AOO", "EIO"},
    3: {"AII", "IAI", "OAO", "EIO"},
    4: {"AEE", "IAI", "EIO"},
}


def parse_categorical(s):
    """'All A are B' / 'No A are B' / 'Some A are not B' / 'c is B' -> (letter, subj, pred).

    A singular statement is treated as universal over a one-member class, which is
    the standard move and is what lets 'S is a M' drive Barbara.

    Terms are canonicalised for articles only. Two terms are the same term when
    their labels match; "men" and "man" do NOT unify, by design.
    """
    t = " ".join(s.strip().split())
    low = t.lower()
    for word, letter in (("some", "I"), ("all", "A"), ("every", "A"), ("no", "E")):
        if low.startswith(word + " "):
            rest = t[len(word) + 1:]
            if " are not " in rest.lower():
                i = rest.lower().index(" are not ")
                return ("O" if letter == "I" else letter, _term(rest[:i]),
                        _term(rest[i + len(" are not "):]))
            for sep in (" are ", " is "):
                if sep in rest.lower():
                    i = rest.lower().index(sep)
                    return (letter, _term(rest[:i]), _term(rest[i + len(sep):]))
            raise ValueError(f"no copula in {s!r}")
    for sep in (" is not ", " is ", " are not ", " are "):
        if sep in low:
            i = low.index(sep)
            letter = "E" if "not" in sep else "A"
            return (letter, _term(t[:i]), _term(t[i + len(sep):]))
    raise ValueError(f"unrecognised categorical proposition: {s!r}")


def classify_categorical(premises, conclusion):
    """['categorical syllogism'] when the mood and figure are valid, else []."""
    if len(premises) != 2:
        return []
    try:
        prems = [parse_categorical(p) for p in premises]
        q_c, subj, pred = parse_categorical(conclusion)
    except ValueError:
        return []

    def terms(p):
        return {p[1], p[2]}

    # The major premise holds the conclusion's predicate, the minor its subject.
    major = next((p for p in prems if pred in terms(p)), None)
    minor = next((p for p in prems if subj in terms(p) and p is not major), None)
    if major is None or minor is None:
        return []
    middles = (terms(major) | terms(minor)) - {subj, pred}
    if len(middles) != 1:
        return []
    middle = middles.pop()
    if middle not in terms(major) or middle not in terms(minor):
        return []

    figure = {(True, True): 1, (False, True): 2,
              (True, False): 3, (False, False): 4}[
        (major[1] == middle, minor[2] == middle)]
    mood = major[0] + minor[0] + q_c
    return ["categorical syllogism"] if mood in VALID_MOODS[figure] else []


# ------------------------------------------------------------------ entry point

def classify(formalization):
    """Name the form of a skeleton. None when nothing matches.

    `formalization` is {"kind": "propositional"|"categorical"|"none",
                        "atoms": {"P": "it rained", ...} or
                        "terms": {"M": "men", ...},
                        "premises": [...], "conclusion": "..."}.

    The mapping is carried for humans and for downstream use; the checker reads
    only the labels, so a wrong gloss cannot change the derived form.

    kind "none" means the argument's force does not come from its shape -- an ad
    hominem is not a bad inference pattern, it is an irrelevant appeal -- so there
    is nothing here to check and the caller keeps the model's own judgement.
    """
    kind = (formalization or {}).get("kind", "none")
    prems = (formalization or {}).get("premises") or []
    concl = (formalization or {}).get("conclusion")
    if kind == "none" or not prems or not concl:
        return None
    try:
        found = (classify_propositional(prems, concl) if kind == "propositional"
                 else classify_categorical(prems, concl) if kind == "categorical"
                 else [])
    except ValueError:
        return None
    return found[0] if len(found) == 1 else None


# ------------------------------------------------------------------ self-tests

def _tests():
    ok = fail = 0

    def check(label, got, want):
        nonlocal ok, fail
        if got == want:
            ok += 1
        else:
            fail += 1
            print(f"  FAIL {label}\n       got  {got!r}\n       want {want!r}")

    # --- parser ------------------------------------------------------------
    check("-> is right associative", parse("P -> Q -> R"),
          ("->", ("atom", "P"), ("->", ("atom", "Q"), ("atom", "R"))))
    check("& binds tighter than |", parse("P | Q & R"),
          ("|", ("atom", "P"), ("&", ("atom", "Q"), ("atom", "R"))))
    check("~ binds tightest", parse("~P & Q"),
          ("&", ("~", ("atom", "P")), ("atom", "Q")))
    check("double negation collapses", parse("~~P"), parse("P"))
    check("parens override precedence", parse("(P | Q) & R"),
          ("&", ("|", ("atom", "P"), ("atom", "Q")), ("atom", "R")))
    for bad in ("P ->", "P Q", "(P", "P & ", "~"):
        try:
            parse(bad)
            check(f"rejects {bad!r}", "accepted", "ValueError")
        except ValueError:
            ok += 1

    # --- each form is recognised -------------------------------------------
    cases = [
        (["P -> Q", "P"],        "Q",      ["modus ponens"]),
        (["P -> Q", "~Q"],       "~P",     ["modus tollens"]),
        (["P -> Q", "Q"],        "P",      ["affirming the consequent"]),
        (["P -> Q", "~P"],       "~Q",     ["denying the antecedent"]),
        (["P -> Q", "Q -> R"],   "P -> R", ["hypothetical syllogism"]),
        (["P | Q", "~P"],        "Q",      ["disjunctive syllogism"]),
        (["P | Q", "~Q"],        "P",      ["disjunctive syllogism"]),
        (["P -> (Q & ~Q)"],      "~P",     ["reductio ad absurdum"]),
        (["P -> Q", "P"],        "P",      ["begging the question"]),
    ]
    for prems, concl, want in cases:
        check(f"{prems} |- {concl}", classify_propositional(prems, concl), want)

    # --- the confusion this module exists to prevent -----------------------
    check("MP is not AC", classify_propositional(["P -> Q", "P"], "Q"), ["modus ponens"])
    check("AC is not MP", classify_propositional(["P -> Q", "Q"], "P"),
          ["affirming the consequent"])
    check("MT is not DA", classify_propositional(["P -> Q", "~Q"], "~P"),
          ["modus tollens"])
    check("DA is not MT", classify_propositional(["P -> Q", "~P"], "~Q"),
          ["denying the antecedent"])

    # --- no skeleton may match two forms -----------------------------------
    for prems, concl, want in cases:
        got = classify_propositional(prems, concl)
        check(f"unambiguous: {prems} |- {concl}", len(got), 1)

    # --- metavariables bind to compound formulas ---------------------------
    check("MP with a compound antecedent",
          classify_propositional(["(A & B) -> C", "A & B"], "C"), ["modus ponens"])
    check("MT with a compound consequent",
          classify_propositional(["A -> (B | C)", "~(B | C)"], "~A"), ["modus tollens"])
    check("premise order does not matter",
          classify_propositional(["P", "P -> Q"], "Q"), ["modus ponens"])

    # --- nonsense is reported as nonsense ----------------------------------
    check("unrelated conclusion", classify_propositional(["P -> Q", "R"], "S"), [])
    check("wrong arity", classify_propositional(["P -> Q"], "Q"), [])

    # --- categorical --------------------------------------------------------
    check("Barbara (AAA-1)",
          classify_categorical(["All men are mortal", "All Greeks are men"],
                               "All Greeks are mortal"), ["categorical syllogism"])
    check("singular subject drives Barbara",
          classify_categorical(["All M are P", "S is a M"], "S is P"),
          ["categorical syllogism"])
    check("articles are ignored",
          classify_categorical(["All M are P", "S is M"], "S is P"),
          ["categorical syllogism"])
    check("inconsistent term labels are rejected, not guessed",
          classify_categorical(["All men are mortal", "Socrates is a man"],
                               "Socrates is mortal"), [])
    check("EIO-1 (Ferio)",
          classify_categorical(["No birds are mammals", "Some pets are birds"],
                               "Some pets are not mammals"), ["categorical syllogism"])
    check("AAA-2 is invalid",
          classify_categorical(["All dogs are animals", "All cats are animals"],
                               "All cats are dogs"), [])
    check("undistributed middle rejected",
          classify_categorical(["All P are M", "All S are M"], "All S are P"), [])

    # --- the entry point ----------------------------------------------------
    check("classify() propositional", classify(
        {"kind": "propositional", "premises": ["P -> Q", "Q"], "conclusion": "P"}),
        "affirming the consequent")
    check("classify() categorical", classify(
        {"kind": "categorical", "terms": {"M": "men", "P": "mortal", "S": "Socrates"},
         "premises": ["All M are P", "S is a M"],
         "conclusion": "S is P"}), "categorical syllogism")
    check("classify() declines kind=none", classify({"kind": "none"}), None)
    check("classify() declines nonsense", classify(
        {"kind": "propositional", "premises": ["P -> Q"], "conclusion": "Z"}), None)
    check("classify() declines unparseable", classify(
        {"kind": "propositional", "premises": ["P ->"], "conclusion": "Q"}), None)
    check("classify() declines empty", classify(None), None)

    # --- no two forms may claim the same skeleton --------------------------
    # Hand-written cases only cover shapes someone thought of. This enumerates a
    # space of arguments and asserts the pattern set never double-matches, which
    # is the property that makes a single derived answer trustworthy.
    space = ["P", "Q", "~P", "~Q"]
    for a, b in itertools.product(list(space), repeat=2):
        space += [f"({a} -> {b})", f"({a} | {b})", f"({a} & {b})"]
    space = sorted(set(space))
    ambiguous = n = 0
    for prems in itertools.combinations(space, 2):
        for concl in space:
            n += 1
            if len(classify_propositional(list(prems), concl)) > 1:
                ambiguous += 1
    check(f"no ambiguity over {n} enumerated arguments", ambiguous, 0)

    print(f"\n{ok} passed, {fail} failed")
    return fail


if __name__ == "__main__":
    import sys
    sys.exit(1 if _tests() else 0)
