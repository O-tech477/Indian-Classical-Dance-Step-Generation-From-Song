"""Constants shared by every stage of the dance pipeline."""
import sys

BONES = ["sp", "lua", "lfa", "rua", "rfa", "lt", "ls", "rt", "rs"]
LSH, RSH, LHP, RHP, LAN, RAN = 11, 12, 23, 24, 27, 28
Q = 4          # planning resolution: quarter beats
N = 200        # samples per adavu cycle
SR = 22050
HOP = 512
STAGE_X = 3.2


def log(*a):
    print(*a, file=sys.stderr)
