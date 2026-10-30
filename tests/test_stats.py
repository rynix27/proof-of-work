from core.stats import gini


def test_perfect_equality_is_zero():
    assert gini([100, 100, 100, 100, 100]) == 0.0


def test_total_monopoly_matches_theoretical_value():
    # for n agents where one has everything, Gini = (n-1)/n
    assert round(gini([0, 0, 0, 0, 500]), 4) == round(4 / 5, 4)


def test_empty_list_does_not_crash():
    assert gini([]) == 0.0


def test_all_zero_does_not_crash_div_by_zero():
    assert gini([0, 0, 0]) == 0.0


def test_negative_balances_are_clamped_to_zero():
    # bankrupt agents shouldn't be able to make the economy look *more*
    # equal by going negative -- they're clamped to 0 contribution
    assert gini([-50, 100, 100]) == gini([0, 100, 100])


def test_result_is_always_between_zero_and_one():
    for values in ([1, 2, 3, 4, 5], [10, 10, 10], [1, 1, 1, 1, 1000]):
        g = gini(values)
        assert 0.0 <= g <= 1.0
