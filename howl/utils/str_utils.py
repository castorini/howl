from distutils.util import strtobool as distutils_strtobool


def strtobool(bool_str: str):
    """Convert a string representation of truth to True or False.
    Wraps disutils.strtobool, please see documentation here: https://docs.python.org/3/distutils/apiref.html
    Quote at time of writing:
    True values are y, yes, t, true, on and 1; false values are n, no, f, false, off and 0.
    Raises ValueError if bool_str is anything else.

    Note: default value for boolean argument is always false
          that we had some difficulties in controlling the flag from our code
          Therefore, we use string-typed argument and convert the value to boolean using strtobool

    Args:
        bool_str: True values are y, yes, t, true, on and 1; false values are n, no, f, false, off and 0.

    Returns:
        bool - Raises ValueError if bool_str is not an accepted string
    """
    return bool(distutils_strtobool(bool_str.lower()))
