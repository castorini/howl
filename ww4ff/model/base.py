MODEL_MAP = dict()


def find_model(name):
    return MODEL_MAP[name]


def model_names():
    return set(MODEL_MAP.keys())


def register_model(name):
    def wrapper(cls):
        MODEL_MAP[name] = cls
        return cls
    return wrapper
