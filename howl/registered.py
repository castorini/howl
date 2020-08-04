from typing import List, Type, TypeVar


T = TypeVar('T', bound='RegisteredObjectBase')


class RegisteredObjectBase:
    registered_map = {}

    def __init_subclass__(cls, name: str = None):
        if name is not None:
            cls.registered_map[name] = cls

    @classmethod
    def registered_names(cls) -> List[str]:
        return list(cls.registered_map.keys())

    @classmethod
    def find_registered_class(cls: Type[T], name: str) -> Type[T]:
        return cls.registered_map[name]
