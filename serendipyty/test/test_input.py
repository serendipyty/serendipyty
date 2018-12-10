import numpy as np
from serendipyty.seismic.input.wavelets import RickerWavelet
import unittest

class TestWavelets(unittest.TestCase):

    def test_upper(self):
        nt = 1001
        dt = 1e-3
        t = np.arange(nt)*dt
        wav = RickerWavelet(t, fc=20, delay=0.05)
        self.assertTrue(wav.wavelet.max() == 1)

if __name__ == '__main__':
    unittest.main()
