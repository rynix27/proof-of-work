"""
Task bank for the economy. Deliberately a single task type (short-answer
logic/trivia questions) per the solo-scope decision: auto-gradable against
a known answer, no need for a judge model, no ambiguity about who "won" a
task. Add more entries any time -- difficulty affects the bounty.
"""
import hashlib
import random
import re
import threading
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


@dataclass
class Task:
    id: int
    prompt: str
    difficulty: str  # "easy" | "medium" | "hard" | "custom"
    bounty: int
    answer: str | None = None
    injected: bool = False  # True for a live "ask the economy" question


_BOUNTY_BY_DIFFICULTY = {"easy": 8, "medium": 15, "hard": 28}

# (prompt, answer, difficulty). Keep answers short and unambiguous so simple
# string-normalize-and-compare grading is reliable -- no judge model needed.
_TASK_BANK = [
    # --- easy ---
    ("What is 17 * 6?", "102", "easy"),
    ("What is the capital of Australia?", "canberra", "easy"),
    ("What is the chemical symbol for gold?", "au", "easy"),
    ("Solve for x: 3x + 7 = 22", "5", "easy"),
    ("What year did the Berlin Wall fall?", "1989", "easy"),
    ("What is the smallest country in the world by area?", "vatican city", "easy"),
    ("How many sides does a dodecagon have?", "12", "easy"),
    ("What is the boiling point of water in Fahrenheit at sea level?", "212", "easy"),
    ("What is 144 divided by 12?", "12", "easy"),
    ("What is the capital of Japan?", "tokyo", "easy"),
    ("How many continents are there?", "7", "easy"),
    ("What is the square root of 169?", "13", "easy"),
    ("What is the chemical symbol for sodium?", "na", "easy"),
    ("What planet is known as the Red Planet?", "mars", "easy"),
    ("How many minutes are in 3 hours?", "180", "easy"),
    ("What is 8 squared?", "64", "easy"),
    ("Who wrote the play 'Romeo and Juliet'?", "shakespeare", "easy"),
    ("What is the freezing point of water in Celsius?", "0", "easy"),
    ("How many sides does a hexagon have?", "6", "easy"),
    ("What is the capital of Canada?", "ottawa", "easy"),

    # --- medium ---
    ("If a train leaves at 3pm and travels for 2 hours 45 minutes, what time does it arrive?", "5:45pm", "medium"),
    ("A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?", "0.05", "medium"),
    ("What is the next prime number after 97?", "101", "medium"),
    ("If you rearrange the letters 'CIFAIPC' you get the name of an ocean. Which?", "pacific", "medium"),
    ("What is the derivative of x^3 with respect to x?", "3x^2", "medium"),
    ("Three friends split a $45 bill equally, then one pays $6 more for dessert. How much did that friend pay total?", "21", "medium"),
    ("What programming paradigm does Haskell primarily follow?", "functional", "medium"),
    ("What is the sum of the interior angles of a hexagon, in degrees?", "720", "medium"),
    ("Which planet has the most moons in our solar system?", "saturn", "medium"),
    ("A recipe needs 3 cups of flour for 12 cookies. How many cups for 20 cookies?", "5", "medium"),
    ("What is 15% of 240?", "36", "medium"),
    ("If today is Wednesday, what day of the week is it 100 days from now?", "friday", "medium"),
    ("What is the value of 2^10?", "1024", "medium"),
    ("A car travels 240 miles using 8 gallons of gas. What is its miles per gallon?", "30", "medium"),
    ("What is the least common multiple of 4 and 6?", "12", "medium"),
    ("In Roman numerals, what number does MCMXCIV represent?", "1994", "medium"),
    ("What is the area of a rectangle with width 7 and height 9?", "63", "medium"),
    ("What is the chemical formula for table salt?", "nacl", "medium"),
    ("How many degrees are in the interior angles of a triangle, total?", "180", "medium"),
    ("What is the median of the numbers 3, 9, 1, 7, 5?", "5", "medium"),
    ("What year was the first iPhone released?", "2007", "medium"),
    ("What is 7 factorial (7!)?", "5040", "medium"),
    ("If a shirt is $40 and is discounted 25%, what is the sale price?", "30", "medium"),
    ("What gas do plants primarily absorb for photosynthesis?", "carbon dioxide", "medium"),
    ("What is the binary representation of the decimal number 10?", "1010", "medium"),

    # --- hard ---
    ("A farmer has chickens and cows totaling 20 heads and 56 legs. How many chickens?", "12", "hard"),
    ("A is twice as old as B. In 10 years, A will be 1.5 times as old as B. How old is B now?", "10", "hard"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?", "5", "hard"),
    ("A rectangle has a perimeter of 30 and a length twice its width. What is its area?", "50", "hard"),
    ("A snail climbs 3 feet up a well each day and slips back 2 feet each night. The well is 10 feet deep. On which day does it reach the top?", "8", "hard"),
    ("What is the sum of all integers from 1 to 100?", "5050", "hard"),
    ("Two trains 120 miles apart travel toward each other at 30 mph and 50 mph. How many hours until they meet?", "1.5", "hard"),
    ("A number is divisible by both 9 and 4. What is the smallest such positive number?", "36", "hard"),
    ("If x + y = 10 and x - y = 4, what is x times y?", "21", "hard"),
    ("In a room of 23 people, is the probability that two share a birthday closer to 10%, 50%, or 90%?", "50%", "hard"),
]

_next_id = 0
_id_lock = threading.Lock()


def _normalize_id():
    """Thread-safe id allocator (injected tasks are created on server threads)."""
    global _next_id
    with _id_lock:
        _next_id += 1
        return _next_id


def reset_ids():
    """Restart task ids at 1 (called at the start of every run so ids are reproducible)."""
    global _next_id
    with _id_lock:
        _next_id = 0


def is_heldout(prompt: str) -> bool:
    """Deterministic ~20% held-out split of the task bank, by prompt hash.
    Held-out prompts are EXCLUDED from training data and used by
    training/evaluate.py, so evaluation never sees training examples."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return digest[0] % 5 == 0


def generate_task(rng: random.Random | None = None, weights: tuple | None = None,
                  bounties: tuple | None = None) -> Task:
    """Pick a bank task. `weights` = (easy, medium, hard) relative weights; when
    given, a difficulty is drawn first, then a task uniformly within it."""
    r = rng or random
    if weights:
        difficulty = r.choices(("easy", "medium", "hard"), weights=weights)[0]
        prompt, answer, difficulty = r.choice([t for t in _TASK_BANK if t[2] == difficulty])
    else:
        prompt, answer, difficulty = r.choice(_TASK_BANK)
    return Task(
        id=_normalize_id(),
        prompt=prompt,
        answer=answer,
        difficulty=difficulty,
        bounty=(dict(zip(("easy", "medium", "hard"), bounties))[difficulty] if bounties
                else _BOUNTY_BY_DIFFICULTY[difficulty]),
    )


CUSTOM_TASK_BOUNTY = 15  # treated as "medium" for economic purposes


def make_custom_task(prompt: str, answer: str = None) -> Task:
    """
    Builds a live, user-injected task for the "ask the economy" feature.
    If `answer` is given, it's graded and paid out exactly like any other
    task. If omitted, the task is an ungraded exhibition round: agents
    still bid and answer for real, but no bounty is paid and no
    correctness is recorded, since there's no ground truth to check
    against. Either way, it's a genuine live call to Nemotron reasoning
    about a question nobody wrote in advance.
    """
    return Task(
        id=_normalize_id(),
        prompt=" ".join(prompt.split())[:500],
        answer=" ".join(answer.split())[:200] if answer else None,
        difficulty="custom",
        bounty=CUSTOM_TASK_BOUNTY if answer else 0,
        injected=True,
    )


_ANSWER_LOOKUP = {prompt: answer for prompt, answer, _ in _TASK_BANK}


def answer_for_prompt(prompt: str) -> str | None:
    """Used only by the fake-model path so offline dev runs produce a
    realistic mix of correct/incorrect outcomes instead of always failing."""
    return _ANSWER_LOOKUP.get(prompt)


_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


def _norm(value) -> str | None:
    """Normalize an answer for comparison; None if it isn't comparable text."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    if not isinstance(value, str):
        return None
    s = value.lower().strip()
    s = s.replace("\u00b2", "^2").replace("\u00b3", "^3").replace("*", "")
    s = "".join(s.split()).replace("$", "").replace(",", "").replace("%", "")
    s = s.strip("\"'`").rstrip(".")
    return s or None


def _as_decimal(s: str):
    if _NUM_RE.match(s):
        try:
            return Decimal(s)
        except InvalidOperation:
            return None
    return None


def grade(task: Task, submitted_answer) -> bool | None:
    """
    True/False for a graded task, or None for an ungraded exhibition task
    (injected question with no known answer) -- callers must treat None as
    "don't touch the economy", not as incorrect.

    Missing, empty, non-text or malformed answers are simply incorrect (False);
    they never raise. Numeric answers compare by value ("5.0" == "5").
    """
    if task.answer is None:
        return None
    expected = _norm(task.answer)
    got = _norm(submitted_answer)
    if expected is None or got is None:
        return False
    if got == expected:
        return True
    e, g = _as_decimal(expected), _as_decimal(got)
    return e is not None and g is not None and e == g
