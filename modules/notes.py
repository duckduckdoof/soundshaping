"""
notes.py

Author: Caleb Scott

---

Descriptions of notes, and their MIDI numbers.

This note system assumes that C3 is MIDI 60.
"""

# IMPORTS
import math
import re

from typing import Union

# CONSTANTS

# Regex for Note notation
NOTE_RE = "(?P<n>[A-G]#?)(?P<oct>-?[0-9])"

# MIDI/freq tuning: 440 Hz.
BASE_FREQ = 440
BASE_MIDI = 69
MIDI_MAX = 128

# 12 note scale lookup
# NOTE: it may be helpful to create a more flexible system for
#   flats/sharps with respect to their musical scales.
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

NOTE_LOOKUP_INV = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11
}

# Octave arrangement
BASE_OCTAVE = -2
STD_OCTAVE = 3

# Scales
MAJ_SCALE = [2, 2, 1, 2, 2, 2, 1]
MIN_SCALE = [2, 1, 2, 2, 1, 2, 2]

# CLASSES
class Note:
    """
    A Note contains information about what can be turned into sound.
    """

    def __init__(
        self,
        repr: Union[int, str],
        r_type: str
    ):
        # Based on the r_type, determine how to build the note.
        # This can either be 'midi' (int) or 'note' (str)
        if r_type == 'midi':
            self.m = int(repr)
            self.n = midi_to_note(self.m)
            self.f = midi_to_freq(self.m)
        elif r_type == 'note':
            self.n = str(repr)
            self.m = note_to_midi(self.n)
            self.f = note_to_freq(self.n)
        else:
            raise Exception("Invalid representation type for a Note.")
        res = re.search(NOTE_RE, self.n)
        if not res:
            raise Exception("Octave not properly calculated in Note creation.")
        self.oct = int(res.group("oct"))

    def __str__(self) -> str:
        return f"{self.n:4s}"

class StdNotes:
    """
    This class generates valid notes on the full MIDI range.
    """

    def __init__(self) -> None:
        self.notes = []

        # Raw range of all midi notes
        for i in range(MIDI_MAX):
            self.notes.append(Note(i, 'midi'))

        # Organized by octave
        self.oct_notes = {}
        for note in self.notes:
            if note.oct not in self.oct_notes:
                self.oct_notes[note.oct] = [note]
            else:
                self.oct_notes[note.oct].append(note)

        # Organized by note string (which includes name+oct)
        self.descr_notes = {n.n: n for n in self.notes}

        # Organized by midi number
        self.midi_notes = {n.m: n for n in self.notes}

    def __getitem__(self, key: str):
        """
        Quick accessor for getting a Note object, given its string.
        """
        return self.descr_notes[key]

    def get_by_notes(self, notes: list) -> list:
        """
        Returns note objects based on their string notation.

        TODO: mark for sharp/flat notations
        """
        return [self[n] for n in notes]

    def get_scale(self, tonic: Note, scale: list = MAJ_SCALE) -> list:
        """
        Given a selected octave, base note (called 'tonic'), and key (maj/min/etc.),
        return all possible notes which describe this scale.

        'key' is a list of relative offsets (e.g. - +2, +2, +1, +2, +2, +2, +1 = maj)

        NOTE we are taking advantage of the fact that our notes are organized by midi
        number: each index is equivalent to taking +/- a half step, or one midi number.
        """
        results = [tonic]
        idx = tonic.m
        for offset in scale:
            idx += offset
            results.append(self.notes[idx])
        return results

    def semitone_len(self, n1: Note, n2: Note) -> int:
        """
        Returns positive distance between two notes.
        """
        return abs(semitone_dist(n1, n2))

    def print_notes(self) -> None:
        current_oct = self.notes[0].oct
        for n in self.notes:
            if n.oct != current_oct:
                current_oct = n.oct
                print()
            print(str(n), end=' ')
        print()

# FUNCTIONS
def midi_to_freq(
    midi_no: int,
    base_freq: int = BASE_FREQ,
    base_midi: int = BASE_MIDI,
    midi_max: int = MIDI_MAX
) -> float:
    """
    Given a midi number, calculate its frequency.
    """
    if midi_no < 0 or midi_no > midi_max:
        return -1.0
    return base_freq * (2 ** ((midi_no - base_midi)/12))

def midi_to_note(
    midi_no: int,
    midi_max: int = MIDI_MAX,
    base_oct: int = BASE_OCTAVE
) -> str:
    """
    Given a midi number, determine the corresponding musical note and its octave.

    This references a table which is used in Logic Studio.
    """
    if midi_no < 0 or midi_no > midi_max:
        raise Exception(f"MIDI number outside max MIDI range [0 - {midi_max}]")
    n_len = len(NOTE_LOOKUP)
    note = midi_no % n_len
    octave = int(((midi_no - note) / n_len) + base_oct)
    return f"{NOTE_LOOKUP[note]}{octave}"

def note_to_midi(note: str) -> int:
    """
    Given a note description (ex: 'A5'), determine its midi number.
    """
    res = re.search(NOTE_RE, note)
    if not res:
        raise Exception("Invalid Note notation.")
    n = NOTE_LOOKUP_INV[res.group("n")]
    base = (int(res.group("oct")) - BASE_OCTAVE) * len(NOTE_LOOKUP_INV)
    return n + base

def note_to_freq(note: str) -> float:
    """
    Given a note description (ex: 'A5'), determine its frequency.
    """
    midi = note_to_midi(note)
    return midi_to_freq(midi)

# These freq -> X methods require high precision in the frequency to get
# back to its corresponding note/midi number. Useful, but not the most helpful.
def freq_to_midi(
    freq: float,
    base_midi: int = BASE_MIDI,
    base_freq: int = BASE_FREQ
) -> int:
    """
    Given a midi number, calculate it's frequency.
    """
    return int(base_midi + 12 * math.log2(freq/base_freq))

def freq_to_note(freq: float) -> str:
    midi = freq_to_midi(freq)
    return midi_to_note(midi)

def semitone_dist(n1: Note, n2: Note) -> int:
    """
    Returns signed distance current note is from another.
    """
    return n2.m - n1.m
