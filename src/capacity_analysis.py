def projected_headroom(capacity_threshold: float, forecast_values):
    if capacity_threshold < 0:
        raise ValueError("Capacity threshold must be nonnegative.")
    return [float(capacity_threshold - value) for value in forecast_values]

def breach_flags(capacity_threshold: float, forecast_values):
    # These are point-forecast threshold flags, not breach probabilities.
    if capacity_threshold < 0:
        raise ValueError("Capacity threshold must be nonnegative.")
    return [bool(value > capacity_threshold) for value in forecast_values]
