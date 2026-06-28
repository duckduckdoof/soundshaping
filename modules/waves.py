"""
waves.py

Author: Caleb Scott

---

Module for describing different wave types.
"""

# IMPORTS
import numpy as np

# CLASSES
class Wave:

    def __init__(self, w_type: str):
        if w_type == 'sin':
            self.f = sin
        elif w_type == 'f_square':
            self.f = f_square
        elif w_type == 'f_triangle':
            self.f = f_triangle
        elif w_type == 'f_saw':
            self.f = f_saw
        elif w_type == 'square':
            self.f = nf_square
        else:
            raise Exception("Wave type not recognized")

# FUNCTIONS

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

# MAIN
if __name__ == 'main':
    pass
