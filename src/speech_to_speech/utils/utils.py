import uuid

import numpy as np


def next_power_of_2(x: int) -> int:
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _generate_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def short_sid(session_id: str) -> str:
    """A compact, human-attributable tag for a session id.

    Our ids are ``"{prefix}_{hex}"`` (e.g. ``session_2f1c…``), so ``[:8]`` would
    just yield the constant prefix. Take the first 8 chars of the random tail
    instead — enough to disambiguate sessions in thread names and log lines."""
    return session_id.rsplit("_", 1)[-1][:8]


def int2float(sound: np.ndarray) -> np.ndarray:
    """
    Taken from https://github.com/snakers4/silero-vad
    """

    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / 32768
    sound = sound.squeeze()  # depends on the use case
    return sound
