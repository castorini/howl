from hashlib import sha256


def sha256_int(value: str):
    m = sha256()
    m.update(value.encode())
    return int(m.hexdigest(), 16)
