from core.control import SimulationControl


def test_starts_unpaused():
    c = SimulationControl()
    assert c.is_paused() is False


def test_pause_and_resume():
    c = SimulationControl()
    c.pause()
    assert c.is_paused() is True
    c.resume()
    assert c.is_paused() is False


def test_set_speed_clamps_to_reasonable_bounds():
    c = SimulationControl()
    c.set_speed(-5)
    assert c.round_delay == 0.1  # clamped up from an invalid negative value
    c.set_speed(1000)
    assert c.round_delay == 30.0  # clamped down from an absurd value
    c.set_speed(5)
    assert c.round_delay == 5


def test_new_generation_invalidates_old_generation():
    c = SimulationControl()
    old_gen = c.start_new_generation()
    new_gen = c.start_new_generation()
    assert old_gen != new_gen
    assert c.should_stop(old_gen) is True  # stale thread should exit
    assert c.should_stop(new_gen) is False  # current thread should keep going


def test_request_stop_affects_current_generation():
    c = SimulationControl()
    gen = c.start_new_generation()
    assert c.should_stop(gen) is False
    c.request_stop()
    assert c.should_stop(gen) is True


def test_new_generation_clears_pause_and_stop():
    c = SimulationControl()
    c.pause()
    c.request_stop()
    gen = c.start_new_generation()
    assert c.is_paused() is False
    assert c.should_stop(gen) is False
