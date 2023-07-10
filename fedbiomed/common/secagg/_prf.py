import hashlib
from math import ceil

from Crypto.Cipher import ChaCha20
from Crypto.Hash import SHA256
from gmpy2 import mpz



class PRF(object):
    _zero = 0
    _nonce = _zero.to_bytes(12, "big")
    security = 256

    def __init__(self, vectorsize, elementsize) -> None:
        super().__init__()
        self.vectorsize = vectorsize
        self.bits_ptxt = elementsize
        self.num_bytes = ceil(elementsize / 8)

    def eval_key(self, key, round):
        if isinstance(key, mpz):
            key = int(key)
        if isinstance(key, int):
            if key >= 2 ** PRF.security:
                mask = 2 ** PRF.security - 1
                key &= mask
            key = key.to_bytes(PRF.security // 8, "big")
        elif not isinstance(key, bytes):
            raise ValueError("seed should be of type either int or bytes")
        round_number_bytes = round.to_bytes(16, 'big')
        c = ChaCha20.new(key=key, nonce=PRF._nonce).encrypt(round_number_bytes)
        c = str(int.from_bytes(c[0:4], 'big') & 0xFFFF)
        return c

    def eval_vector(self, seed):

        c = ChaCha20.new(key=seed, nonce=PRF._nonce)
        data = b"secr" * self.vectorsize
        return c.encrypt(data)
