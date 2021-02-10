import json

from pydantic import BaseSettings
from pathlib import PosixPath


JSON_PRIMITIVES = {int, float, complex, list, tuple, range, str, bytes, bytearray, memoryview, set, frozenset, dict,
                   None, bool}
SERIALIZABLE_TYPES = {PosixPath}


def gather_dict(dataclass, keys_to_ignore = []):
    items = dataclass.dict().items() if isinstance(dataclass, BaseSettings) else dataclass.__dict__.items()
    data_dict = dict()
    for k, v in items:
        if k in keys_to_ignore:
            continue
        elif type(v) in JSON_PRIMITIVES or v is None:
            data_dict[k] = v
        elif type(v) in SERIALIZABLE_TYPES:
            data_dict[k] = str(v)
        else:
            data_dict[k] = gather_dict(v)
    return data_dict


def prettify_dataclass(dataclass):
    return json.dumps(gather_dict(dataclass), indent=2)
