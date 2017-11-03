"""
MIT License

Copyright (c) 2017 Kairos Aerospace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Implementation of a generic airborne detection system. Subclass of the DetectionMethod class.
Supports 3 different probability of detection curves:
  "legacy": A hard-coded point-by-point probability curve.
  "exponential": A parameterized exponential curve
  "exp_erf": A parameterized erf-based S-curve that is guaranteed to go to zero probability at
  zero leak rate.

The parameterized models are especially useful for sensitivity calculations.

"""
import numpy as np
from scipy.special import erf

from DetectionModules.abstract_detection_method import DetectionMethod
from DetectionModules import helper_functions
from GeneralClassesFunctions.simulation_functions import sample_wr, set_kwargs_attrs
from feast_constants import *


class AD(DetectionMethod):
    def __init__(self, simulation_time, gas_field, **kwargs):
        """
        AD is the model of how an airborne detector detects leaks.
        Inputs:
           gas_field    a gas_field object (Defined in feast_classes)
           time         a time object (Defined in feast_classes)
           kwargs       optional input dicitionary that will override default parameters
        """
        super().__init__(simulation_time, gas_field)
        # -------------- LeakSurveyor default properties ---------------------
        # Detection
        self.name = "AD"
        # Some generic detection probabilities, to be replaced with actual model
        # used only in the "legacy" mode.
        self.detection_sizes = np.array([0, 1, 2, 3, 4, 1000]) * mcfpd_to_gps
        self.detection_prob = np.array([0, 0.5, 0.75, 0.9, 0.99, 1])
        self.detection_sizes = np.sort(self.detection_sizes)
        self.detection_prob = np.sort(self.detection_prob)
        # Survey
        self.survey_interval = 30  # days
        # Money
        self.lifetime = 10 * 365

        # Detection probability models
        self.detection_models = {
            "exponential": self.get_exp_probabilities,
            "legacy": self.get_legacy_probabilities,
            "exp_erf": self.get_exp_erf_probabilities
        }
        if "detection_model_name" not in kwargs:
            kwargs.update({"detection_model_name": "legacy"})
        if "inst_params" not in kwargs:
            kwargs.update({"inst_params": {}})

        self.inst_params_index = None
        self.survey_interval_index = None
        # -------------- Override default parameters with kwargs --------------
        set_kwargs_attrs(self, kwargs, only_existing=False)

        # Calculate capital costs vector
        self.capital_0 = None
        self.capital = np.zeros(simulation_time.n_timesteps)  # dollars
        helper_functions.replacement_cap(simulation_time, self)

        #  The finding np.costs must be calculated as leaks are found. The vector is
        #  initialized here.
        self.find_cost = [0] * simulation_time.n_timesteps

    def __str__(self):
        msg = "AD: Model: {}  Interval (d): {}  ".format(self.detection_model_name,
                                                         self.survey_interval)
        if "exp_erf" in self.detection_model_name:
            msg += "Center (Mscf/day): {}  Width: {}  Scale: {}".format(self.inst_params[
                                                                            "center"] / mcfpd_to_gps,
                                                                        self.inst_params[
                                                                            "width"] / mcfpd_to_gps,
                                                                        self.inst_params[
                                                                            "scale"] / mcfpd_to_gps)

        return msg

    def detection(self, time, gas_field, atm):
        """
        Determines which leaks are detected at each timestep
        Inputs:
            time:        object of type Time defining the time variables in the simulation
            gas_field:  object of type GasField defining gas field parameters
            atm:        object of type Atmosphere defining wind speed, direction and atmospheric
                        stability
        """
        self.null_detection(time, gas_field)
        if self.leaks.n_leaks < 0:
            print("WARN: why are there less than zero leaks")
        # Periodically repair all detected leaks
        if time.current_time % self.survey_interval < time.delta_t:
            # print("AD Flight: Day {}".format(time.current_time))
            # print("num_leaks", self.leaks.n_leaks)
            # print("len flux", len(self.leaks.flux))
            if self.leaks.n_leaks < 0:
                print("WARN: why are there less than zero leaks")
                self.leaks.n_leaks = len(self.leaks.flux)  # TODO still a hack
            flux_nonzero = self.leaks.flux[:self.leaks.n_leaks]

            detected = np.zeros(self.leaks.n_leaks, dtype=int)

            detection_probs = self.get_detection_probabilities(flux_nonzero)

            finding_luck = np.random.rand(self.leaks.n_leaks)
            detected[finding_luck <= detection_probs] = 1
            detected_ndxs = np.where(detected)[0]
            # print("det ndx", detected_ndxs)
            # print("det type", detected.dtype)
            num_found = np.sum(detected)
            if num_found > 0:
                self.repair_cost[time.time_index] += \
                    sum(sample_wr(gas_field.repair_cost_dist.repair_costs, len(detected_ndxs)))
                self.leaks_found.extend(flux_nonzero[detected_ndxs])
                # Delete found leaks
                self.leaks.delete_leaks(detected_ndxs)
                # msg = "AD found {} leaks on day {:.1f}! Remaining leaks: {} Largest unfound: {}"
                # print(msg.format(num_found, time.current_time, self.leaks.n_leaks,
                #                  np.max(self.leaks.flux)))

    def get_detection_probabilities(self, fluxes):
        """
        Access function to detection probability models.
        Either point this at the one you really like, or switch based on some flag.
        :param fluxes: 1D array of fluxes
        :return: probs: 1D array of detection probabilities
        """
        probs = self.detection_models[self.detection_model_name](fluxes, self.inst_params)
        # probs = self.get_legacy_probabilities(fluxes)
        # probs = self.get_exp_probabilities(fluxes)
        # probs = self.get_exp_erf_probabilities(fluxes, center=30 * mcfpd_to_gps, width=50 *
        # mcfpd_to_gps)
        return probs

    def get_legacy_probabilities(self, fluxes, inst_params):
        detection_probs = np.zeros_like(fluxes)
        for i in range(1, len(self.detection_sizes)):
            p_lower = self.detection_prob[i - 1]
            p_upper = self.detection_prob[i]
            size_lower = self.detection_sizes[i - 1]
            size_upper = self.detection_sizes[i]
            leaks_in_size_braket = (size_lower < fluxes) & \
                                   (fluxes <= size_upper)
            # print("len det probs", len(detection_probs))
            # print("len flux", len(self.leaks.flux))
            # print("subset 1", detection_probs[leaks_in_size_braket])
            # print("subset 2", self.leaks.flux[leaks_in_size_braket])
            detection_probs[leaks_in_size_braket] = \
                p_lower + (p_upper - p_lower) * \
                          (fluxes[leaks_in_size_braket] - size_lower) / \
                          (size_upper - size_lower)

        detection_probs[fluxes > max(self.detection_sizes)] = 1

        return detection_probs

    def get_exp_erf_probabilities(self, fluxes, inst_params):
        """
        The erf function is an s-curve, but it's not guaranteed to drop to zero at leak-size = 0.
        So, multiplying by an appropriate exponential (exp[-k/x], where k is a scale and x is
        leak size) forces the function down to zero for small leaks.

        :param fluxes: np.array of measured fluxes in gps
        :param inst_params: dict with the following:
          center: flux with a ~50/50 chance of being detected, in gps
          width: steepness of the s-curve, in gps
          scale: Scale size of the exponential that makes sure we have zero probability at
        zero leak rate
        :return: probabilities
        """
        prob = np.exp(-inst_params["scale"] / fluxes) * self.erf_dist(fluxes,
                                                                      inst_params["center"],
                                                                      inst_params["width"])

        return prob

    def get_exp_probabilities(self, fluxes, inst_params):
        """
        The exp version is a pure exponential. While guaranteed to go to zero at zero leak size,
        it tends to have higher detection probabilities at medium leak sizes, as compared to the
        exp-erf, when the two are made to match at large leaks.

        I don't have any a priori reason to favor one of these over another at first. As we gather
        more data in the field we may prefer one of them to another.

        :param fluxes: np.array of fluxes
        :param scale: scale size of the exponential
        :return: probabilities
        """
        prob = 1 - np.exp(-fluxes / inst_params["scale"])

        return prob

    def erf_dist(self, x, center, width):
        """
        Given a center and width, calculate the probability of detection of a leak of size x
        :param center: The leak size that has a 50/50 detection probability
        :param width: A scale width indicating how steep the probability rise is.
        :return: probability
        """

        return 0.5 + (erf((x - center) / width) / 2)
