import copy
import numpy as np
import abc

from GeneralClassesFunctions.simulation_functions import set_kwargs_attrs
from GeneralClassesFunctions.simulation_functions import sample_wr


class DetectionMethod(metaclass=abc.ABCMeta):
    """
    DetectionMethod is an abstract super class that defines the form required for all detection
    methods
    """

    def __init__(self, simulation_time, gas_field, notes=None, **kwargs):
        """
        Inputs:
            time         a time object (Defined in simulation_classes)
            gas_field    a gas_field object (Defined in simulation_classes)
            notes        a description of the object created
            kwargs       optional input dict that will override default parameters
        """
        self.name = ""
        self.notes = notes
        self.null_repaired = []
        self.repair_cost = [0] * simulation_time.n_timesteps
        self.capital = [0] * simulation_time.n_timesteps
        self.find_cost = [0] * simulation_time.n_timesteps
        self.leaks_found = []
        # leaks will be updated throughout simulations. initial_leaks should remain constant, so a
        # copy is needed.
        self.leaks = copy.deepcopy(gas_field.initial_leaks)
        self.count_found = []
        self.leakage = []
        # Set all attributes defined in kwargs, regardless of whether they already exist
        set_kwargs_attrs(self, kwargs, only_existing=False)

    def __str__(self):
        msg = "{}".format(self.name)

    @abc.abstractmethod
    def detection():
        pass

    def null_detection(self, time, gas_field):
        """
        Every detection method shares the same null_detection method defined here
        Inputs:
            time         a time object (Defined in simulation_classes)
            gas_field    a gas_field object (Defined in simulation_classes)
        """
        # print("{}: Null detection: Chance of fixing each leak: {}".format(
        #     self.name, gas_field.null_repair_rate * time.delta_t))
        n_repaired = np.random.poisson(len(self.leaks.flux) * gas_field.null_repair_rate
                                       * time.delta_t)
        if n_repaired > len(self.leaks.flux):
            n_repaired = len(self.leaks.flux)

        if n_repaired > 0:
            # msg = "{}: Randomly found {} leaks on day {}! Remaining leaks: {}"
            # print(msg.format(self.name, n_repaired, time.current_time, len(self.leaks.flux) -
            #                  n_repaired))
            index_repaired = np.random.choice(n_repaired, n_repaired, replace=False)
        else:
            index_repaired = []
        for ind in index_repaired:
            self.null_repaired.append(self.leaks.flux[ind])
        self.leaks.delete_leaks(index_repaired)
        self.repair_cost[time.time_index] += sum(sample_wr(gas_field.repair_cost_dist.repair_costs,
                                                           n_repaired))
