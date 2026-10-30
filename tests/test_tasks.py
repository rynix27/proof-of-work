from core import tasks


def test_generate_task_returns_known_bank_entry():
    task = tasks.generate_task()
    assert task.prompt
    assert task.answer
    assert task.difficulty in ("easy", "medium", "hard")
    assert task.bounty > 0


def test_generate_task_ids_are_unique_and_increasing():
    t1 = tasks.generate_task()
    t2 = tasks.generate_task()
    assert t2.id > t1.id


def test_grade_exact_match():
    task = tasks.Task(id=1, prompt="what is 2+2", answer="4", difficulty="easy", bounty=8)
    assert tasks.grade(task, "4") is True


def test_grade_case_insensitive():
    task = tasks.Task(id=1, prompt="capital of japan", answer="tokyo", difficulty="easy", bounty=8)
    assert tasks.grade(task, "Tokyo") is True
    assert tasks.grade(task, "TOKYO") is True


def test_grade_ignores_whitespace():
    task = tasks.Task(id=1, prompt="q", answer="vatican city", difficulty="easy", bounty=8)
    assert tasks.grade(task, "  vatican   city  ") is True


def test_grade_ignores_dollar_and_comma():
    task = tasks.Task(id=1, prompt="q", answer="0.05", difficulty="medium", bounty=15)
    assert tasks.grade(task, "$0.05") is True
    task2 = tasks.Task(id=2, prompt="q", answer="5040", difficulty="medium", bounty=15)
    assert tasks.grade(task2, "5,040") is True


def test_grade_wrong_answer_is_false():
    task = tasks.Task(id=1, prompt="q", answer="42", difficulty="easy", bounty=8)
    assert tasks.grade(task, "43") is False


def test_answer_for_prompt_looks_up_known_question():
    task = tasks.generate_task()
    assert tasks.answer_for_prompt(task.prompt) == task.answer


def test_answer_for_prompt_returns_none_for_unknown_question():
    assert tasks.answer_for_prompt("this is not a real question") is None


def test_task_bank_has_reasonable_variety():
    # regression guard: catches an accidental truncation of the task bank
    assert len(tasks._TASK_BANK) >= 40
    difficulties = {t[2] for t in tasks._TASK_BANK}
    assert difficulties == {"easy", "medium", "hard"}


def test_grade_ignores_percent_sign():
    task = tasks.Task(id=1, prompt="q", answer="50%", difficulty="hard", bounty=28)
    assert tasks.grade(task, "50") is True
    assert tasks.grade(task, "50%") is True


def test_every_bank_answer_is_gradable_against_itself():
    # guards against a malformed entry whose own answer can't match itself
    for prompt, answer, difficulty in tasks._TASK_BANK:
        t = tasks.Task(id=0, prompt=prompt, answer=answer, difficulty=difficulty, bounty=1)
        assert tasks.grade(t, answer) is True
