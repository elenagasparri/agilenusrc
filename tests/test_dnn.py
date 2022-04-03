"""Unit test for the DataSet.
"""

import unittest
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
if sys.flags.interactive:
    plt.ion()
    
from agilenusrc.blazardnn import BlazarDNN

class testBlazarDnn(unittest.TestCase):
    '''Unit test for the BlazarDnn module.
    '''
    
    def test_rescale(self, n=200.):
        '''Test that rescaleData function returns an
        nd-array with value in axis=0 between [0,1].
        '''
        d=np.linspace(1,100,n)
        rd = np.reshape(d, (10,10,2))
        run = BlazarDnn()
        data_norm = run.rescaleData(rd)
        
        self.assertTrue(data_norm[:,:,0].min()==0)
        self.assertTrue(data_norm[:,:,0].max()==1)
        
if __name__ == '__main__':
    unittest.main(exit=not sys.flags.interactive)
    
    


