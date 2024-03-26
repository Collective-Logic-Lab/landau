# test_landau.py
#
# Bryan Daniels
# 2023/4/15
#
# Run tests on fitting the Landau distribution to data.
#

from landau import landauAnalysis as la
import numpy as np
import scipy.stats
import unittest

TEST_DATA_1D = np.transpose([[ 1,2,2,3,6,7,7,9 ]])
TEST_DATA_ONE_PEAK = np.array(
      [-1.20989661,  1.37743574, -0.34309834,  3.60503794,
       -1.11522358, -2.90860392,  1.59253442, -0.30426874,
        1.11779486,  2.38415287, -2.58817198,  2.5608723 ,
       -0.89261858,  2.29564895, -0.61865523, -4.9529401 ])
TEST_DATA_TWO_PEAK = np.array(
      [-15.56813387,  15.80659403,  16.68259997,  16.17744904,
        16.47473291, -16.10998222, -17.38832699,  16.92766729,
        16.85658502,  16.06551564, -16.61063726, -16.60344865,
       -16.41819715, -16.33074274, -17.91711983,  17.9554448 ])
TEST_DATA_BIASED = np.array(
      [-0.02037413,  0.12069606, -0.17127188, -0.25660994,
       0.43111868, 0.43111868,  0.66078961, -1.10812355,
       0.05060321, -1.10812355,-0.04987375,  0.23914247,
       -1.26340974,  0.28082133,  0.02085288,0.47896815,
       -1.42144096,  0.89363424, -1.52357434,  1.16865798,
        1.16865798,  1.15377847, -1.59293993,  0.16168133,
       -1.58984572, 0.14147   ,  0.13278455,  0.21457888,
        0.82803159,  0.20835128, 1.2415132 ,  0.52058969,
       -1.63953704, -1.64729957,  0.22907616, 1.47981246,
        1.00511421, -1.83816787,  1.0111071 ,  0.70428524,
        0.88229831,  0.7978728 ,  0.93043928,  0.55239871,
        1.25535428, 1.07447751, -2.31846138,  0.70383624,
        1.15678024,  1.15678024, 0.61196692, -2.12126737,
        0.94908863,  0.94908863,  0.94816161, 0.95038748,
        0.91860909,  0.95038748, -2.35532129,  1.13758168,
        1.26102567,  1.04077159,  0.65682665,  1.12008028,
        0.73337574, 0.77164676,  1.59734924,  0.84611661,
        1.37283304,  1.66251743, 1.13409161,  1.71496109,
        1.95126029,  1.89009535,  0.7751016 , 1.7975893 ,
        1.39454858,  1.4291981 ,  1.42637294, -1.4106421 ,
        1.0441909 , -1.4106421 ,  1.25055274, -1.45018201,
        1.25055274, 0.7791704 , -1.44471435,  1.44415336,
        1.44415336,  1.3011054 , 1.31011943,  1.35273371,
        1.13284199,  1.17602663,  1.26274349, 1.18027044,
        1.11296915,  1.23315889,  1.13676067,  1.07257044])

class TestLandau(unittest.TestCase):
    
    def test_landau_1D(self):
        """
        Test fitting Landau distribution to 1D data
        """
        
        fitting_data = la.landauAnalysis(TEST_DATA_1D)
        
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
                               
    def test_landau_one_peak_unbiased(self):
        """
        Test fitting unbiased Landau distribution to data
        with one peak.
        
        These data come from the all-to-all simulation with
        interaction strength mu = 0., runIndex = 18,
        nComponents = 91, projected onto the first principal
        component.
        
        Here we compare the results to what we get using
        the Mathematica minimizer.
        """
        result = la.maximize_landau_log_likelihood(
                 TEST_DATA_ONE_PEAK,abs_hmax=1e-20)
                 
        fit_c,_,fit_d,fit_numu = result.x
        ll = result.fun
        
        self.assertAlmostEqual(0.40755588661074127,fit_c,
                               places=5)
        self.assertAlmostEqual(0.24287142185587904,fit_d,
                               places=5)
        self.assertAlmostEqual(-0.18754865966284,fit_numu,
                               places=5)
        self.assertAlmostEqual(-0.16968415748600307,ll,
                               places=5)
        
    def test_landau_two_peaks_unbiased(self):
        """
        Test fitting unbiased Landau distribution to data
        with two peaks.
        
        These data come from the all-to-all simulation with
        interaction strength mu = 0.02197802197802198,
        runIndex = 18, nComponents = 91, projected onto the
        first principal component.
        
        Here we compare the results to what we get using
        the Mathematica minimizer.  In this case,
        Mathematica finds a slightly better solution, which
        increases the log-likelihood less than 1e-5, but
        changes the fit parameters c and d on the order of
        1e-2.
        """
        result = la.maximize_landau_log_likelihood(
                 TEST_DATA_TWO_PEAK,abs_hmax=1e-20)
                 
        fit_c,_,fit_d,fit_numu = result.x
        ll = result.fun
        
        self.assertAlmostEqual(-315.44178219,fit_c,
                               places=1)
        self.assertAlmostEqual(314.43525984,fit_d,
                               places=1)
        self.assertAlmostEqual(-0.00357573,fit_numu,
                               places=4)
        self.assertAlmostEqual(-40.40928411,ll,
                               places=5)
                               
    def test_landau_two_peaks_biased(self):
        """
        Test fitting biased Landau distribution to data
        with two peaks.
        
        These data come from BoolODE simulated data.
        """
        result = la.maximize_landau_log_likelihood(
                 TEST_DATA_BIASED)
                 
        fit_c,fit_h,fit_d,fit_numu = result.x
        ll = result.fun
        
        self.assertAlmostEqual(-2.6139158368699342,fit_c,
                               places=5)
        self.assertAlmostEqual(-1.0739188587744732,fit_h,
                               places=5)
        self.assertAlmostEqual(1.5595897919529675,fit_d,
                               places=5)
        self.assertAlmostEqual(-1.1366408704329083,fit_numu,
                               places=5)
        self.assertAlmostEqual(-33.79734705695188,ll,
                               places=5)
                               

class TestLandauHelpers(unittest.TestCase):

    def test_normalizationZ(self):
        """
        Test calculation of normalization factor Z
        """
        Z1 = la.normalizationZ(1,1,1,1)
        self.assertAlmostEqual(Z1,2.042460133912278)
        
        # harder case that requires maxorder > 100
        Z2 = la.normalizationZ(0.003,-300,0.1,300)
        self.assertAlmostEqual(Z2,1.3993104423514305e+33)

    def test_gaussian(self):
        """
        Test calculation of Gaussian likelihoods
        """
        # compare our gaussian pdf to scipy.stats's
        x = 0.1
        loc = 2
        sigma = 5
        lpdf = la.GaussianDistributionLogPDF(x,
            loc,1./sigma)
        lpdf2 = scipy.stats.norm.logpdf(x,
            loc=loc,scale=np.sqrt(sigma))
        self.assertAlmostEqual(lpdf,lpdf2)
