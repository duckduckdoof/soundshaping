"""
notes.py

Author: Caleb Scott

---

Descriptions of notes, and their MIDI numbers.
"""

# CONSTANTS

# MIDI/freq tuning: 440 Hz.
BASE_FREQ = 440
BASE_MIDI = 69
MIDI_MAX = 128

# 12 note scale lookup
NOTE_LOOKUP = {
    0:  "C",
    1:  "C#",
    2:  "D",
    3:  "D#",
    4:  "E",
    5:  "F",
    6:  "F#",
    7:  "G",
    8:  "G#",
    9:  "A",
    10: "A#",
    11: "B"
}

# Octave arrangement
BASE_OCTAVE = -2
STD_OCTAVE = 3

# FUNCTIONS

# MIDI conversions
def midi_to_freq(
    midi_no: int,
    base_freq: int = BASE_FREQ,
    base_midi: int = BASE_MIDI
) -> float:
    """
    Given a midi number, calculate its frequency.
    """
    if midi_no < 0 or midi_no > MIDI_MAX:
        return -1.0
    return base_freq * (2 ** ((midi_no - base_midi)/12))

def midi_to_note(midi_no: int) -> tuple:
    """
    Given a midi number, determine the corresponding musical note and its octave.

    This references a table which is used in Logic Studio.
    """
    if midi_no < 0 or midi_no > MIDI_MAX:
        raise Exception(f"MIDI number outside max MIDI range [0 - {MIDI_MAX}")
    n_len = len(NOTE_LOOKUP)
    note = midi_no % n_len
    octave = int(((midi_no - note) / n_len) + BASE_OCTAVE)
    return f"{NOTE_LOOKUP[note]}", octave

# CLASSES
class Note:
    """
    A Note contains information about what can be turned into sound.
    """

    def __init__(
        self,
    ):
        pass
