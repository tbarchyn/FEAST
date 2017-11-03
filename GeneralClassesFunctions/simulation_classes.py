"""
feast_classes stores the basic classes used by FEAST.
Additional classes are stored in DetectionModules directory and leak_objects.
"""
import numpy as np
import pickle

from .leak_class_functions import leak_objects_generator as leak_obj_gen
from .leak_class_functions import make_leaks
import GeneralClassesFunctions.simulation_functions as sf

# Constants:
GROUND_TEMP = 300  # K
PRESSURE = 101325  # Pa


class GasField:
    """
    GasField accommodates all data that defines a gas field at the beginning of a simulation.
    """

    def __init__(self, initial_leaks=None, null_repair_rate=None, **kwargs):
        """
        Input params:
            initial_leaks     The set of leaks that exist at the beginning of the simulation
            null_repair_rate  The rate at which leaks are repaired in the Null process
                              (repairs/leak/day)
            kwargs            All attributes defined in the kwargs section below
        """

        # -------------- Attributes that can be defined with kwargs --------------
        # Type of distribution used to generate leak sizes
        self.dist_type = "bootstrap"
        # Rate at which new leaks are produced
        self.leak_production_rate = 1e-5  # new leaks per component per day
        # Number of valves and connectors per well (736659 total components/1138 wells in the Fort
        # Worth study)
        self.components_per_site = 650
        # Number of wells to be simulated
        self.site_count = 100
        # Maximum number of wells to be surveyed with a single capital investment
        self.max_count = 6000
        # Driving distance between wells
        self.site_spacing = 700  # m
        # Concentration of wells
        self.well_density = 2  # wells per km^2
        # Square root of the area over which a leak may be found per well. Based on satellite
        # imagery.
        self.well_length = 10  # m
        # Maximum leak height
        self.h0_max = 5  # m
        # Plume temperature
        self.t_plume = 300  # K
        # Update any attributes defined by kwargs
        # ---------- Leak distribution specific params - Can be set with kwargs
        # Path to a LeakData object file
        if "dist_type" in kwargs:
            self.dist_type = kwargs.pop("dist_type")
        if self.dist_type in {"bootstrap", "power_law", "lognormal"}:
            self.leak_data_path = "fort_worth_leaks.p"
        else:
            self.leak_data_path = None

        if "max_leak" in kwargs:
            max_leak = kwargs.pop("max_leak")
        else:
            max_leak = 100  # g/s

        if self.dist_type == "power_law":
            self.power_law_stats = {
                "alpha": 1.75,
                "min": 0.30808,  # g/s
                "max": max_leak  # g/s
            }
            # self.power_law_alpha = 1.75
            # self.power_law_min = 0.30808
        else:
            self.power_law_stats = None
            # self.power_law_alpha = None
            # self.power_law_min = None
        if self.dist_type == "lognormal":
            # print("lognormal:: leak_data_path: {}".format(self.leak_data_path))
            self.lognormal_stats = {
                "min": 0.30808,  # g/s
                "small_frac": 99.5,  # percent
                "mu": 10 ** 1.75,  # from Frankenberg et al
                "sigma": 0.55
            }
        else:
            self.lognormal_stats = None

        sf.set_kwargs_attrs(self, kwargs)

        # -------------- Calculated parameters --------------
        # Define functions and parameters related to leaks
        self.leak_params, self.leaks_per_well = \
            leak_obj_gen(self.dist_type, self.leak_data_path)
        # Define the number of leaks in each well site
        self.leaks_in_well = np.random.poisson(self.leaks_per_well, self.site_count)
        n_leaks = int(sum(self.leaks_in_well))
        # Define the initial set of leaks
        if initial_leaks is None:
            self.initial_leaks = make_leaks(n_leaks, self)  # g/s
        else:
            self.initial_leaks = initial_leaks
        # Total number of components in the simulation
        self.component_count = self.components_per_site * self.site_count
        # Null repair rate
        if null_repair_rate is None:
            # leaks repaired per leak per day
            self.null_repair_rate = self.leak_production_rate * self.component_count / n_leaks
        else:
            self.null_repair_rate = null_repair_rate
        # Distribution of leak costs
        self.repair_cost_dist = pickle.load(open(
            "InputData/DataObjectInstances/fernandez_leak_repair_costs_2006.p", "rb"))

    def __str__(self):
        msg = "GasField:: Type: {}".format(self.dist_type)
        msg += "  Site count: {}".format(self.site_count)
        return msg


# FinanceSettings stores all parameters relating to economic calculations
class FinanceSettings:
    def __init__(self, gas_price=2E-4, discount_rate=0.08):
        self.gas_price = gas_price  # dollars/gram (2e-4 $/g=$5/mcf methane at STP)
        self.discount_rate = discount_rate


class Atmosphere:
    """
    Defines atmosphere variables for use in plume simulations
    """

    def __init__(self, timesteps, wind_speed_path="arpae_wind.p",
                 wind_direction_path="fort_worth_wind.p", **kwargs):
        """
        Inputs
            timesteps           number of timesteps in the simulation
            wind_speed_path     path to a wind data object
            wind_direction_path path to a wind data object (may or may not be the same as
                                wind_speed_path)
        """
        self.wind_speed = []
        self.wind_direction = []
        self.stab_class = []
        self.r_y = []
        self.r_z = []
        speed_data = pickle.load(open("InputData/DataObjectInstances/" + wind_speed_path, "rb"))
        dir_data = pickle.load(open("InputData/DataObjectInstances/" + wind_direction_path, "rb"))
        a = np.array([927, 370, 283, 707, 1070, 1179])
        l = np.array([0.102, 0.0962, 0.0722, 0.0475, 0.0335, 0.022])
        q = np.array([-1.918, -0.101, 0.102, 0.465, 0.624, 0.700])
        k = np.array([0.250, 0.202, 0.134, 0.0787, .0566, 0.0370])
        p = np.array([0.189, 0.162, 0.134, 0.135, 0.137, 0.134])
        self.wind_speed = np.zeros(timesteps)
        self.wind_direction = np.zeros(timesteps)
        self.stab_class = np.zeros(timesteps)
        self.ground_temp = np.ones(timesteps) * GROUND_TEMP
        self.a_temp = self.ground_temp - 20
        self.pressure = np.ones(timesteps) * PRESSURE
        # emissivities of the ground and air
        self.e_a = np.ones(timesteps) * 0.1
        self.e_g = np.ones(timesteps) * 0.5
        # Stability classes are chosen randomly with equal probability, subject to constraints
        # based on wind speed.  Stability classes 5 and 6 are never chosen because they rarely
        # occur during the day.
        sc = None
        if "stability_class" in kwargs:
            sc = kwargs.pop("stability_class")
            print("Using stability class {}".format(sc))

        for ind in range(0, timesteps):
            # Set up a deterministic set of wind speeds for testing purposes
            # Note: Wind direction only matters for Distributed Detector (DD) type
            if sc is not None:
                self.stab_class[ind] = sc
                self.wind_direction[ind] = 0
                if sc < 2:
                    self.wind_speed[ind] = 1.4
                elif sc < 3:
                    self.wind_speed[ind] = 2.4
                elif sc < 5:
                    self.wind_speed[ind] = 3.7
                else:
                    self.wind_speed[ind] = 7.2

            else:  # use the bootstrap wind speeds
                self.wind_speed[ind] = np.random.choice(speed_data.wind_speed)
                self.wind_direction[ind] = np.random.choice(dir_data.wind_direction)
                if self.wind_speed[ind] < 2:
                    self.stab_class[ind] = np.random.randint(2)
                elif self.wind_speed[ind] < 3:
                    self.stab_class[ind] = np.random.randint(3)
                elif self.wind_speed[ind] < 5:
                    self.stab_class[ind] = np.random.randint(1, 4)
                    self.stab_class[ind] = np.random.randint(1, 4)
                else:
                    self.stab_class[ind] = np.random.randint(2, 4)
        sf.set_kwargs_attrs(self, kwargs)
        self.a, self.l, self.q = np.zeros(timesteps), np.zeros(timesteps), np.zeros(timesteps)
        self.k, self.p = np.zeros(timesteps), np.zeros(timesteps)
        for ind in range(0, timesteps):
            self.a[ind] = a[int(self.stab_class[ind])]
            self.l[ind] = l[int(self.stab_class[ind])]
            self.q[ind] = q[int(self.stab_class[ind])]
            self.k[ind] = k[int(self.stab_class[ind])]
            self.p[ind] = p[int(self.stab_class[ind])]


class Time:
    """
    Instances of the time class store all time related information during a simulation
    """

    def __init__(self, delta_t=3650 / 4000, end_time=10 * 365, current_time=0, time_index=0):
        """
        Inputs:
        delta_t                 length of one timestep (days)
        end_time                length of the simulation (days)
        current_time            current time in a simulation (days)
        """
        self.n_timesteps = int(end_time / delta_t + 1)
        self.end_time = end_time
        self.delta_t = delta_t
        self.current_time = current_time
        self.time_index = time_index

    def __str__(self):
        msg = "Time:: end_time: {}".format(self.end_time)
        msg += "  delta_t: {}".format(self.delta_t)
        msg += "  n_timesteps: {}".format(self.n_timesteps)

        return msg


class Results:
    """
    Object in which to save results
    """

    def __init__(self, time, gas_field, tech_dict, leak_list, no_repair_leakage, atm, econ_set,
                 new_leaks, n_leaks):
        """
        Inputs:
        time                    Time object
        gas_field               GasField object
        tech_dict               dict of detection methods and associated data
        leak_list               Leak object, containing all leaks ever created for this sim. This
                                 is independent of the currently-existing-leaks maintained for each
                                 instrument.
        no_repair_leakage       list of leakage in a scenario with no repairs
        atm                     Atmosphere object
        econ_set          Economic settings defined for the simulation
        new_leaks               List of Leak generated at each time step. Apparently redundant data
                                 with leak_list, except that it's broken out by timestep.
        n_leaks                 A 1-D list of current leak count
                                 of length (num detectors) x (num timesteps). To be useful, it has
                                 to be reshaped into a 2D array.
        """
        self.time = time
        self.gas_field = gas_field
        self.tech_dict = tech_dict
        self.leak_list = leak_list
        self.no_repair_leakage = no_repair_leakage
        self.atm = atm
        self.econ_settings = econ_set
        self.new_leaks = new_leaks
        self.n_leaks = n_leaks

    def __str__(self):
        msg = "Result: tech_dict: {}\n".format([tech for tech in self.tech_dict.keys()])
        for tech in self.tech_dict.keys():
            msg += "  {}\n".format(self.tech_dict[tech])
        msg += "  {}\n".format(self.gas_field)
        msg += "  {}\n".format(self.time)
        msg += "  {}\n".format(self.leak_list)
        return msg
