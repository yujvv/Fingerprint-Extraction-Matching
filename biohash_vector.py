import numpy as np
import random
import hashlib

biometric_template = np.zeros(128)
# User-specific Tokenized Random Numbers (TRNs)
trn = [random.randint(0, 1) for i in range(128)]

# computes the BioHash by performing a bitwise multiplication between the biometric template and TRN, and then taking the result modulo 2 to get the binary bits
def biohash(biometric_template, trn):
    biohash = np.zeros(128, dtype=int)
    for i in range(128):
        biohash[i] = (biometric_template[i] * trn[i]) % 2
    return biohash

# To encrypt the BioHash
def encrypt(biohash):
    sha = hashlib.sha256()
    sha.update(biohash.tobytes())
    encrypted = np.frombuffer(sha.digest(), dtype=int)
    encrypted = encrypted[:128]  # Truncate to match BioHash length
    return encrypted

print("---------biometric_template-------", biometric_template)

print("--------trn--------", trn)

biohash_ = biohash(biometric_template, trn)
res = encrypt(biohash_)

print("--------biohash_--------", biohash_)
print("--------res--------", res)