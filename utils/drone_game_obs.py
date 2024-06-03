import numpy as np


def binary_to_decimal(o_bin):
    o_dec = int("".join(str(int(b)) for b in o_bin)[::-1], base=2)
    return o_dec


def decimal_to_binary(o_dec, width):
    o_bin = [int(b) for b in np.binary_repr(o_dec, width=width)][::-1]

    return o_bin
