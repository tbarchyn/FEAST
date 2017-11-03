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

HY is a hybrid model, representing a tiered approach. It consists of applying both an airborne
detector and an OGI on different schedules; generally the AD is operated more frequently,
with an occasional complete OGI survey.

The Cost functionality has not yet been implemented.
"""
import numpy as np

from GeneralClassesFunctions.simulation_classes import Time
import DetectionModules as DM
from DetectionModules.abstract_detection_method import DetectionMethod
from DetectionModules import helper_functions
from GeneralClassesFunctions.simulation_functions import sample_wr, set_kwargs_attrs
from feast_constants import *

import logging


class HY(DetectionMethod):
    def __init__(self, simulation_time, gas_field, **kwargs):
        """
        HY is a hybrid model including an airborne detector and an ogi, used together on
        different revisit frequencies.
        Inputs:
           gas_field    a gas_field object (Defined in feast_classes)
           time         a time object (Defined in feast_classes)
           kwargs       optional input dicitionary that will override default parameters
        """
        super().__init__(simulation_time, gas_field)
        # -------------- LeakSurveyor default properties ---------------------
        # Detection
        self.name = "HY"
        # Survey
        self.ad_survey_interval = 90  # days
        self.ogi_survey_interval = 730  # days
        self.ogi_distance = 10  # m
        # Money
        self.lifetime = 10 * 365

        # Detection probability models
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

        log = logging.getLogger("HYBRID")
        self.null_detection(time, gas_field)
        if self.leaks.n_leaks < 0:
            print("WARN: why are there less than zero leaks")
        # Periodically repair all detected leaks
        ad_count = 0
        ogi_count = 0
        todays_finds_flux = []
        if time.current_time % self.ad_survey_interval < time.delta_t:
            log.debug(
                "Day {:.1f}, performing survey with AD piece of hybrid. Largest existing: {"
                "}".format(
                    time.current_time,
                    np.max(self.leaks.flux)))
            if self.leaks.n_leaks < 0:
                print("WARN: why are there less than zero leaks")
                self.leaks.n_leaks = len(self.leaks.flux)  # TODO still a hack

            # Make an AD with the current set of leaks, and run a detection on it.
            # print("Creating ephemeral AD with {} leaks".format(len(self.leaks.flux)))
            ephemeral_time = Time(current_time=time.current_time)
            ephmeral_AD = DM.airborne_detector.AD(ephemeral_time, gas_field, leaks=self.leaks,
                                                  detection_model_name=self.detection_model_name,
                                                  inst_params=self.inst_params,
                                                  survey_interval=self.ad_survey_interval)
            # print(" Resulting detector has {} leaks".format(len(ephmeral_AD.leaks.flux)))
            ephmeral_AD.detection(ephemeral_time, gas_field, atm)
            if len(ephmeral_AD.leaks_found) > 0:
                # log.debug("  Hyb-AD found these leaks: {}".format(ephmeral_AD.leaks_found))
                self.leaks_found.extend(ephmeral_AD.leaks_found)
            self.leaks = ephmeral_AD.leaks  # ephmeral_AD.leaks has had the found ones deleted
            ad_count = len(ephmeral_AD.leaks_found)
            todays_finds_flux.append(ephmeral_AD.leaks_found)

        if time.current_time % self.ogi_survey_interval < time.delta_t:
            log.debug("Day {:.1f}, performing survey with OGI piece of hybrid. Largest existing: "
                      "{}".format(
                time.current_time,
                np.max(self.leaks.flux)))
            if self.leaks.n_leaks < 0:
                # print("WARN: why are there less than zero leaks")
                self.leaks.n_leaks = len(self.leaks.flux)  # TODO still a hack

            # Make an OGI with the current set of leaks, and run a detection on it.
            # print("Creating ephemeral OGI with {} leaks".format(len(self.leaks.flux)))
            ephemeral_time = Time(current_time=time.current_time, time_index=time.time_index)
            ephemeral_OGI = DM.ir.MIR(ephemeral_time, gas_field,
                                      leaks=self.leaks,
                                      survey_interval=self.ogi_survey_interval,
                                      distance=self.ogi_distance,
                                      name="MIR-HYB")
            # print(" Resulting detector has {} leaks".format(len(ephemeral_OGI.leaks.flux)))
            ephemeral_OGI.detection(ephemeral_time, gas_field, atm)
            # print("MIR-HYB: {}".format(ephemeral_OGI))
            if len(ephemeral_OGI.leaks_found) > 0:
                # print(ephemeral_OGI.leaks_found)
                self.leaks_found.extend(ephemeral_OGI.leaks_found)
            self.leaks = ephemeral_OGI.leaks
            ogi_count = len(ephemeral_OGI.leaks_found)
            todays_finds_flux.append(ephemeral_OGI.leaks_found)

        if ad_count > 0 or ogi_count > 0:
            log.debug("Day {:.1f}: Found {} leaks >= {}. Largest remaining: {}".format(
                time.current_time,
                ad_count + ogi_count,
                np.min(todays_finds_flux),
                np.max(self.leaks.flux)))

            # TODO does not calculate costs
