import numpy as np

from GeneralClassesFunctions import simulation_classes
from DetectionModules.airborne_detector import AD
from feast_constants import *


def test_legacy_AD():
    np.random.seed(2001)
    inst_params = {}  # legacy has no parameters.

    time = simulation_classes.Time()
    gas_field = simulation_classes.GasField()
    test_ad = AD(time, gas_field)

    test_fluxes = np.array([5, 10, 20, 50, 100, 120, 2000]) * mcfpd_to_gps

    probs = test_ad.get_legacy_probabilities(test_fluxes, inst_params=inst_params)

    truth_probs = [0.99001, 0.99006, 0.990161, 0.990462, 0.990964, 0.991165, 1.]

    np.testing.assert_array_almost_equal(probs, truth_probs)


def test_exp_probs():
    np.random.seed(2001)
    inst_params = {"scale": 20 * mcfpd_to_gps}

    time = simulation_classes.Time()
    gas_field = simulation_classes.GasField()
    test_ad = AD(time, gas_field, detection_model_name="exponential")

    test_fluxes = np.array([5, 10, 20, 50, 100, 120, 2000]) * mcfpd_to_gps

    probs = test_ad.get_exp_probabilities(test_fluxes, inst_params=inst_params)

    truth_probs = [0.221199, 0.393469, 0.632121, 0.917915, 0.993262, 0.997521, 1.]
    np.testing.assert_array_almost_equal(probs, truth_probs)


def test_exp_erf_probs():
    np.random.seed(2001)
    inst_params = {
        "center": 35 * mcfpd_to_gps,
        "width": 21 * mcfpd_to_gps,
        "scale": 3 * mcfpd_to_gps
    }

    time = simulation_classes.Time()
    gas_field = simulation_classes.GasField()
    test_ls = AD(time, gas_field)

    test_fluxes = np.array([5, 10, 20, 50, 100, 120, 2000]) * mcfpd_to_gps

    probs = test_ls.get_exp_erf_probabilities(test_fluxes, inst_params)
    truth_probs = [0.011896, 0.034174, 0.134452, 0.79465, 0.97044, 0.97531, 0.998501]
    np.testing.assert_array_almost_equal(probs, truth_probs)
