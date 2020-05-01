import json


JSON_PRIMITIVES = {int, float, complex, list, tuple, range, str, bytes, bytearray, memoryview, set, frozenset, dict,
                   None, bool}


def gather_dict(dataclass):
    data_dict = dict()
    for k, v in dataclass.__dict__.items():
        data_dict[k] = v if type(v) in JSON_PRIMITIVES else gather_dict(v)
    return data_dict


def prettify_dataclass(dataclass):
    return json.dumps(gather_dict(dataclass), indent=2)