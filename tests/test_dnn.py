"""Unit test for the Bidirectional LSTM neural network build in order to classifying blazars.
"""

import unittest
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
if sys.flags.interactive:
    plt.ion()


