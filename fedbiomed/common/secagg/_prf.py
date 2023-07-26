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

    def eval_key(self, key: bytes, round: int):
        round_number_bytes = round.to_bytes(16, 'big')
        c = ChaCha20.new(key=key, nonce=PRF._nonce).encrypt(round_number_bytes)
        # the output is a 16 bytes string, pad it to 32 bytes
        # TODO fix it, I don't know if it is correct to pad with zeros
        c = c + b'\x00' * 16
        return c

    def eval_vector(self, seed):
        c = ChaCha20.new(key=seed, nonce=PRF._nonce)
        data = b"secr" * self.vectorsize
        return c.encrypt(data)
