from src.capacity_analysis import projected_headroom, breach_flags

def test_capacity_scenario_helpers():
    assert projected_headroom(100, [90, 110]) == [10.0, -10.0]
    assert breach_flags(100, [90, 110]) == [False, True]
