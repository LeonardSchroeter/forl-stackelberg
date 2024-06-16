import numpy as np


def binary_to_decimal(o_bin):
    o_dec = int("".join(str(int(b)) for b in o_bin)[::-1], base=2)
    return o_dec


def decimal_to_binary(o_dec, width):
    o_bin = [int(b) for b in np.binary_repr(o_dec, width=width)][::-1]

    return o_bin


def coord_to_repr(coord, base=21):
    repr = 0
    for (i, o) in enumerate(coord[::-1]):
        repr += o * (base**i)
    return repr


def repr_to_coord(repr, base=21, width=8):
    coord = []
    while repr > 0:
        o = repr % base
        coord.insert(0, o)
        repr //= base
    while len(coord) < width:
        coord.insert(0, 0)
    return coord


if __name__ == "__main__":
    coord = [8, 6, 2, 4]
    base=11
    repr = coord_to_repr(coord, base=base)
    print(repr)
    coord_decoded = repr_to_coord(repr, base=base, width=len(coord))
    print(coord_decoded)
