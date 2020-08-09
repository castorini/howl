import os

import coloredlogs


__all__ = []


coloredlogs.install(level=os.environ.get('HOWL_LOG_LEVEL', 'INFO'),
                    fmt='%(asctime)s [%(levelname)s] %(module)s: %(message)s')
