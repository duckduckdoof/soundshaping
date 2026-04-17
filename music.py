"""
music.py

author: Caleb Scott

---

I stink at music -- so why not write programs to understand them better?
"""

# IMPORTS

from typing import Self

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

# CONSTANTS

# Notes

BASE_FREQ = 440
BASE_MIDI = 69
MIDI_MAX = 128

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

BASE_OCTAVE = -2
STD_OCTAVE = 3

# Scales

MAJ_SCALE = [
    2, 2, 1, 2, 2, 2, 1
]

MIN_SCALE = [
    2, 1, 2, 2, 1, 2, 2
]

# Time information

BPM_DEFAULT = 60

# Sampling

SAMPLE_RATE = 44100

sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 2

# CLASSES

class Note:
    """
    A Note contains information about what can be played.
    """

    def __init__(self, midi_no: int) -> None:
        self.m = midi_no
        self.n, self.oct = midi_to_note(midi_no)
        self.f = midi_to_freq(midi_no)

    def semitone_dist(self, note: Self) -> int:
        """
        Returns signed distance current note is from another.
        """
        return note.m - self.m

    def __str__(self) -> str:
        return f"{self.n:2s}[{self.oct:2d}]"

class ADSREnvelope:
    """
    Class for modifying the amplitude/level of some sound using a classic ADSR envelope.

    Designed to be paired with the Sound class, to modify generated sounds.

    Durations for each phase is expressed as a fraction in [0,1] space.
    The total durations for each phase must add to 1.
    """

    def __init__(self, 
                atk_dur: float,
                atk_height: float,
                dec_dur: float,
                dec_height: float,
                sus_dur: float,
                rel_dur: float
            ):

        self.atk_dur = atk_dur
        self.atk_height = atk_height
        self.dec_dur = dec_dur
        self.dec_height = dec_height
        self.sus_dur = sus_dur
        self.rel_dur = rel_dur

class Sound:
    """
    Sound data to be played.
    You can create this information directly, or use Note information.

    This sound can be parameterized by any wave function, so
    long as it returns an np.ndarray
    """

    def __init__(self, 
            freq: int, 
            duration: float, 
            amplitude: float, 
            wave_func,
            adsr: ADSREnvelope = None,
            sample_rate: int = SAMPLE_RATE):
        self.freq = freq
        self.duration = duration
        self.amplitude = amplitude
        self.wave_func = wave_func
        self.sample_rate = sample_rate
        self.n_samples = int(self.sample_rate * self.duration)
        self.adsr = adsr

    def gen_sound_data(self) -> np.ndarray:
        """
        Instead of playing the sound directly, generate the sound
        data to then be played on something.

        Helpful if you want to generate and join sound data,
        and then play it.
        """
        t_points = np.linspace(0, self.duration, self.n_samples, False)

        # Amplitude is determined by 'amplitude' parameter (global scaling)
        # and the provided adsr envelope
        waveform = self.amplitude * np.multiply(
                self.gen_adsr(), 
                self.wave_func(2 * np.pi * self.freq * t_points)
        )
        return waveform

    def gen_adsr(self) -> np.ndarray:
        """
        If an adsr is provided, generate amplitude data.
        """
        base = np.ones(self.n_samples)
        if not self.adsr:
            return base

        # Attack
        atk_len = int(self.adsr.atk_dur * self.n_samples)
        base[0: atk_len] = np.linspace(0.0, self.adsr.atk_height, atk_len, False)

        # Decay
        dec_len = int(self.adsr.dec_dur * self.n_samples)
        dec_offset = atk_len + dec_len
        base[atk_len: dec_offset] = np.linspace(self.adsr.atk_height, self.adsr.dec_height, dec_len, False)

        # Sustain
        sus_len = int(self.adsr.sus_dur * self.n_samples)
        sus_offset = dec_offset + sus_len
        base[dec_offset: sus_offset] = self.adsr.dec_height * np.ones(sus_len)

        # Release
        rel_len = int(self.adsr.rel_dur * self.n_samples)
        base[sus_offset:] = np.linspace(self.adsr.dec_height, 0.0, rel_len, False)

        return base


class Timing:
    """
    We can play sound data, but for how long? How do we align a sequence/harmony of notes onto
    a single timeline? That's what this class is for!
    """

    def __init__(self, bpm: int = BPM_DEFAULT):
        pass

class StdNotes:
    """
    This class generates valid notes on the full MIDI range.
    """

    def __init__(self) -> None:
        self.notes = []

        # Raw range of all midi notes
        for i in range(MIDI_MAX):
            self.notes.append(Note(i))

        # Organized by octave
        self.oct_notes = {}
        for note in self.notes:
            if not note.oct in self.oct_notes:
                self.oct_notes[note.oct] = [note]
            else:
                self.oct_notes[note.oct].append(note)

        # Organized by note string (which includes name+oct)
        self.descr_notes = {str(n.n)+str(n.oct): n for n in self.notes}

        # Organized by midi number
        self.midi_notes = {n.m for n in self.notes}

    def get_by_notes(self, notes: list, octaves: list) -> list:
        """
        Returns note objects based on their string notation.

        TODO: mark for sharp/flat notations
        """
        results = []
        for octave in octaves:
            results.extend([n for n in self.oct_notes[octave] if n.n in notes])
        return results

    def get_scale(self, tonic: Note, key: list) -> list:
        """
        Given a selected octave, base note (called 'tonic'), and key (maj/min/etc.),
        return all possible notes which describe this scale.

        'key' is a list of relative offsets (e.g. - +2, +2, +1, +2, +2, +2, +1 = maj)

        NOTE we are taking advantage of the fact that our notes are organized by midi
        number: each index is equivalent to taking +/- a half step, or one midi number.
        """
        results = [tonic]
        idx = tonic.m
        for offset in key:
            idx += offset
            results.append(self.notes[idx])
        return results

    def semitone_len(self, n1: Note, n2: Note) -> int:
        """
        Returns positive distance between two notes.
        """
        return abs(n1.semitone_dist(n2))

    def print_notes(self) -> None:
        current_oct = self.notes[0].oct
        for n in self.notes:
            if n.oct != current_oct:
                current_oct = n.oct
                print()
            print(str(n), end=' ')
        print()

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
        return "", -10
    n_len = len(NOTE_LOOKUP)
    note = midi_no % n_len
    octave = int(((midi_no - note) / n_len) + BASE_OCTAVE)
    return f"{NOTE_LOOKUP[note]}", octave

# Useful frequency utilities

def get_harmonics(fundamental: int, harmonics: list) -> list:
    """
    Given a fundamental frequency, return the full
    list of harmonic frequencies described by the 'harmonics' list.

    The list is a list of positive integers.
    """
    return [fundamental * h for h in harmonics]

def rel_freq(n1: Note, n2: Note) -> float:
    """
    Given two notes, return their frequency ratio f2/f1.

    This can be achieved two ways:
    1. f2/f1 (f2 >= f1)
    2. 2**|(n2.midi - n1.midi)|/12, assuming a MIDI configuration tuned at 440Hz
    """
    return 2**(abs(n2.m - n1.m)/12)

# Waveforms

def sin(x):
    return np.sin(x)

# Waveforms (other than sine)
# These are fourier approximations of the follwing wave shapes:

def f_square(x, n: int = 100):
    c = 4/np.pi
    return c * sum([np.sin(x * (2*k - 1))/(2*k - 1) for k in range(1, n+1)])

def f_triangle(x, n: int = 100):
    c = 8/(np.pi**2)
    return c * sum([((-1)**k) * np.sin(x * (2*k - 1))/((2*k - 1)**2) for k in range(1, n+1)])

def f_saw(x, n: int = 100):
    c = 2/np.pi
    return c * sum([((-1)**k) * np.sin(x * k)/k for k in range(1, n+1)])

# Non-fourier descriptions

def nf_square(x):
    return np.sign(sin(x))

# Visuals & Playing

def plot_sound(sound_data: np.ndarray):
    """
    Given generated sound data, plot it!
    """
    plt.plot(sound_data)
    plt.show()

def play(sound_data: np.ndarray):
    """
    Plays sound data generated from a numpy array.
    """
    sd.play(sound_data)
    sd.wait()

# MAIN
if __name__ == "__main__":
    pass
