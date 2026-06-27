"""
sound.py

Author: Caleb Scott

---

Module for sound information and playing sounds.
"""

# IMPORTS
from modules.waves import Wave
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# CONSTANTS
SAMPLE_RATE = 44100
NUM_CHANNELS = 2

sd.default.samplerate = SAMPLE_RATE
sd.default.channels = NUM_CHANNELS

# CLASSES
class ADSREnvelope:
    """
    Class for modifying the amplitude/level of some sound using a classic ADSR envelope.

    Designed to be paired with the Sound class, to modify generated sounds.

    Durations for each phase is expressed as a fraction in [0,1] space.
    The total durations for each phase must add to 1.
    """

    def __init__(
        self,
        atk_dur: float,
        atk_height: float,
        dec_dur: float,
        dec_height: float,
        sus_dur: float,
        rel_dur: float
    ) -> None:
        self.atk_dur = atk_dur
        self.atk_height = atk_height
        self.dec_dur = dec_dur
        self.dec_height = dec_height
        self.sus_dur = sus_dur
        self.rel_dur = rel_dur

class Harmonics:
    """
    Class for specifying the harmonics of a sound, given a base frequency.

    NOTE: This should be reworked.

    It is related to the harmonics function below.
    """

    def __init__(self, harmonics: list[float], levels: list[float]) -> None:
        self.freqs = harmonics
        self.base = harmonics[0]
        self.levels = levels

class Sound:
    """
    Sound data to be played.
    You can create this information directly, or use Note information.

    This sound can be parameterized by any wave function, so
    long as it returns an np.ndarray
    """

    def __init__(
        self,
        freq: int,
        duration: float,
        amplitude: float,
        wave: Wave,
        adsr: ADSREnvelope | None = None,
        sample_rate: int = SAMPLE_RATE
    ) -> None:
        self.freq = freq
        self.duration = duration
        self.amplitude = amplitude
        self.wave = wave
        self.sample_rate = sample_rate

        # Might help to clamp the duration to a minimum, since sample rate is finite.
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
                self.wave.f(2 * np.pi * self.freq * t_points)
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

# Useful frequency utilities
def get_harmonics(fundamental: int, harmonics: list) -> list:
    """
    Given a fundamental frequency, return the full
    list of harmonic frequencies described by the 'harmonics' list.

    The list is a list of positive integers.
    """
    return [fundamental * h for h in harmonics]

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
if __name__ == 'main':
    pass
