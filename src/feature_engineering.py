def make_supervised_features(*args, **kwargs):
    raise NotImplementedError(
        "Implement only after reporting frequency and forecast-time feature availability are documented. "
        "For irregular data, row lags are not calendar-day lags."
    )
