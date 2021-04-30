from hashlib import sha256

from howl.data.common.metadata import AudioClipMetadata


def sha256_int(value: str):
    m = sha256()
    m.update(value.encode())
    return int(m.hexdigest(), 16)


class Sha256Splitter:
    def __init__(self, target_pct: int):
        self.target_pct = target_pct

    def __call__(self, x: AudioClipMetadata) -> bool:
        return (sha256_int(str(x.path)) % 100) < self.target_pct
