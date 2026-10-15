def fit_baseline_models(*args, **kwargs):
    raise NotImplementedError("Implement after selecting a defensible time frequency.")

def fit_statistical_models(*args, **kwargs):
    raise NotImplementedError("Implement after checking regularity and series length.")

def fit_machine_learning_models(*args, **kwargs):
    raise NotImplementedError("Implement after leakage-safe features and validation are defined.")
