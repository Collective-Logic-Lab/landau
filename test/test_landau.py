# test_landau.py
#
# Bryan Daniels
# 2023/4/15
#
# Run tests on fitting the Landau distribution to data.
#

from landau import landauAnalysis
import numpy as np
import unittest

TEST_DATA_1D = np.transpose([[ 1,2,2,3,6,7,7,9 ]])

class TestLandau(unittest.TestCase):
    
    def test_landau_1D(self):
        """
        Test fitting Landau distribution to 1D data
        """
        
        fitting_data = landauAnalysis.landauAnalysis(TEST_DATA_1D)
        
        # check mean
        self.assertAlmostEqual(np.mean(TEST_DATA_1D),
                               fitting_data['mu'])
        
        # check fitting parameters
        self.assertAlmostEqual(-4.70707629,
                               fitting_data['cList'][0])
        self.assertAlmostEqual(3.89792329,
                               fitting_data['dList'][0])
        self.assertAlmostEqual(0.22487831,
                               fitting_data['nuMuList'][0])
                               
        # check likelihood calculations
        self.assertAlmostEqual(-2.53539044,
                               fitting_data['llList'][0])
        self.assertAlmostEqual(-2.99133933,
                               fitting_data['bicDiffList'][0])
