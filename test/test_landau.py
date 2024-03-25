# test_landau.py
#
# Bryan Daniels
# 2023/4/15
#
# Run tests on fitting the Landau distribution to data.
#

from landau import landauAnalysis
import numpy as np
import scipy.stats
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

class TestLandauHelpers(unittest.TestCase):

    def test_normalizationZ(self):
        """
        Test calculation of normalization factor Z
        """
        Z1 = landauAnalysis.normalizationZ(1,1,1,1)
        self.assertAlmostEqual(Z1,2.042460133912278)

    def test_gaussian(self):
        """
        Test calculation of Gaussian likelihoods
        """
        # compare our gaussian pdf to scipy.stats's
        x = 0.1
        loc = 2
        sigma = 5
        lpdf = landauAnalysis.GaussianDistributionLogPDF(x,
            loc,1./sigma)
        lpdf2 = scipy.stats.norm.logpdf(x,
            loc=loc,scale=np.sqrt(sigma))
        self.assertAlmostEqual(lpdf,lpdf2)
