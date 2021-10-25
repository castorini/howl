from hashlib import sha256

from howl.data.common.metadata import AudioClipMetadata


def sha256_int(data: str):
    """Apply SHA256 hashing and get integer value

    Args:
        data: data to encode

    Returns:
        SHA256 hash as an integer value
    """
    hash_obj = sha256()
    hash_obj.update(data.encode())
    return int(hash_obj.hexdigest(), 16)


class Sha256Splitter:
    """Splits given audio clip based on the given configuration"""

    def __init__(self, target_pct: int):
        """Initialize Sha256Splitter

        Args:
            target_pct: if the hash value is less than target_pct, it will be returned with True
        """
        self.target_pct = target_pct

    def __call__(self, metadata: AudioClipMetadata) -> bool:
        """Split given audio clip

        Args:
            metadata: metadata of the audio clip

        Returns:
            True if the hash value of the path is less than target_pct
        """
        return (sha256_int(str(metadata.path)) % 100) < self.target_pct
