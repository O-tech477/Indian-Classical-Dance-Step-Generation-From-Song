"""Hinge limits for knees and elbows, on whole cycles and on single key poses."""
import numpy as np


def enforce_anatomical_limits(C):
    """Enforce human biomechanical hinge constraints on knees and elbows.

    In 2D frontal projection of Bharatanatyam (Araimandi stance):
      - Left thigh (lt=5) to left shin (ls=6): knee flexes inward towards body midline.
        Inward flexion delta in [-5°, 150°]. Clamps backward hyperextension / inverted knee.
      - Right thigh (rt=7) to right shin (rs=8): knee flexes inward towards body midline.
        Inward flexion delta in [-150°, 5°]. Clamps backward hyperextension.
      - Elbows: forearm flexion cannot invert past 160° relative to upper arm.
    """
    out = np.array(C, float, copy=True)
    for i in range(len(out)):
        # Left leg: lt=5, ls=6
        dl = (out[i, 6] - out[i, 5] + 180.0) % 360.0 - 180.0
        if dl < -5.0:
            out[i, 6] += (-5.0 - dl)
        elif dl > 150.0:
            out[i, 6] += (150.0 - dl)

        # Right leg: rt=7, rs=8
        dr = (out[i, 8] - out[i, 7] + 180.0) % 360.0 - 180.0
        if dr > 5.0:
            out[i, 8] += (5.0 - dr)
        elif dr < -150.0:
            out[i, 8] += (-150.0 - dr)

        # Left elbow: lua=1, lfa=2
        del_l = (out[i, 2] - out[i, 1] + 180.0) % 360.0 - 180.0
        if abs(del_l) > 160.0:
            out[i, 2] += (np.sign(del_l) * 160.0 - del_l)

        # Right elbow: rua=3, rfa=4
        del_r = (out[i, 4] - out[i, 3] + 180.0) % 360.0 - 180.0
        if abs(del_r) > 160.0:
            out[i, 4] += (np.sign(del_r) * 160.0 - del_r)
    return out


def enforce_pose_limits(p):
    """Enforce anatomical limits on a single pose dictionary."""
    if not isinstance(p, dict):
        return p
    r = dict(p)
    if "lt" in r and "ls" in r:
        dl = (r["ls"] - r["lt"] + 180.0) % 360.0 - 180.0
        if dl < -5.0:
            r["ls"] += (-5.0 - dl)
        elif dl > 150.0:
            r["ls"] += (150.0 - dl)
    if "rt" in r and "rs" in r:
        dr = (r["rs"] - r["rt"] + 180.0) % 360.0 - 180.0
        if dr > 5.0:
            r["rs"] += (5.0 - dr)
        elif dr < -150.0:
            r["rs"] += (-150.0 - dr)
    if "lua" in r and "lfa" in r:
        del_l = (r["lfa"] - r["lua"] + 180.0) % 360.0 - 180.0
        if abs(del_l) > 160.0:
            r["lfa"] += (np.sign(del_l) * 160.0 - del_l)
    if "rua" in r and "rfa" in r:
        del_r = (r["rfa"] - r["rua"] + 180.0) % 360.0 - 180.0
        if abs(del_r) > 160.0:
            r["rfa"] += (np.sign(del_r) * 160.0 - del_r)
    return r
