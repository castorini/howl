import argparse
import enum


__all__ = ['ArgumentParserBuilder', 'opt', 'OptionEnum']


def _make_parser_setter(option, key):
    def fn(value):
        option.kwargs[key] = value
        return option
    return fn


class ArgumentParserOption(object):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __iter__(self):
        return iter((self.args, self.kwargs))

    def __getattr__(self, item):
        if item == 'kwargs':
            return self.kwargs
        if item == 'args':
            return self.args
        if item == 'postprocess':
            return self.postprocess
        return _make_parser_setter(self, item)


opt = ArgumentParserOption


class ArgumentParserBuilder(object):
    def __init__(self, **init_kwargs):
        self.parser = argparse.ArgumentParser(**init_kwargs)
        self.option_map = dict()

    def add_options(self, *options: ArgumentParserOption):
        for args, kwargs in options:
            self.parser.add_argument(*args, **kwargs)
        return self.parser


class OptionEnum(enum.Enum):
    pass
