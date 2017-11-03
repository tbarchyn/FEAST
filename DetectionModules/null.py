from DetectionModules.abstract_detection_method import DetectionMethod
from GeneralClassesFunctions.simulation_functions import set_kwargs_attrs


class Null(DetectionMethod):
    """
    This class specifies a null detection method. It includes a detection method and several
    parameters.
    This method represents the likelihood that normal oilfield operations stumble across a leak.
    It should be instantiated with a Time object that has a delta_t representing how often a
    person visits that well.
    """

    def __init__(self, simulation_time, gas_field, **kwargs):
        """
        Inputs:
              gas_field    a gas_field object (Defined in feast_classes)
              time         a time object (Defined in feast_classes)
              kwargs       optional input dictionary that will override default parameters
        """
        DetectionMethod.__init__(self, simulation_time, gas_field)
        # The null repair rate defaults to the same setting as the gas field, but can be set to a
        # different value.
        self.name = "Null"
        self.repair_rate = gas_field.null_repair_rate

        # -------------- Process details: None --------------
        self.survey_interval = 1
        # -------------- Financial Properties --------------
        # Capital costs are zero in this module.
        self.capital = [0] * simulation_time.n_timesteps  # dollars

        # maintenance costs are zero in the null module
        self.maintenance = [0] * simulation_time.n_timesteps  # $

        # Find cost is zero in the null module
        self.find_cost = [0] * simulation_time.n_timesteps  # $

        # Set attributes defined by kwargs. Only set attributes that already exist.
        set_kwargs_attrs(self, kwargs, only_existing=True)

    def __str__(self):
        msg = "Null: Survey interval (d): {}   Repair rate: {}".format(self.survey_interval,
                                                                       self.repair_rate)
        return msg

    def detection(self, simulation_time, gas_field, atm):
        """
        The null detection method is simply the null detection method defined in the super class
        DetectionMethod
        Inputs:
            simulation_time        an object of type Time (defined in feast_classes) containing
                        the relevant current time and stepping information for the overall sim

            gas_field   an object of type GasField (defined in feast_classes)
            atm         an object of type Atmosphere (defined in feast_classes)
                        Note: atm is not currently used by this method, but it is accepted as an
                        argument for consistency with other detection methods
        """
        if simulation_time.current_time % self.survey_interval < simulation_time.delta_t:
            DetectionMethod.null_detection(self, simulation_time, gas_field)
