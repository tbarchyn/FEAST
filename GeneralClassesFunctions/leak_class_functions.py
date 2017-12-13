"""
Leak data, leak distribution properties, and leak objects are created in this module
"""
import pickle
import numpy as np
from feast_constants import *

# Constants:
g = 9.8  # g is the strength of gravity [m/s^2]
RHO_AIR = 1225  # density of air [g/m^3]
RHO_METHANE = 681  # density of methane at atmospheric pressure [g/m^3]


class Leak:
    """
    Stores a list of leaks
    """

    def __init__(self, x_pos=(), y_pos=(), z_pos=(), flux=(),
                 f_one_third=(), leaks_detected=(), capacity=0):
        """
        Inputs:
        x_pos               East-west position of leak with respect to the center of the well (m)
        y_pos               North-south position of leak with respect to the center o the well (m)
        z_pos               Altitude of leak with respect to the ground (m)
        flux                leak size (g/s)
        f_one_third         A plume dispersion factor
        leaks_detected      Binary value to save whether the leak has been detected or not (1 if
                            detected, 0 otherwise)
        capacity            Expected total number of leaks to be stored in this instance of Leak
                            (allows for faster extend method)
        """
        if f_one_third is () and flux is not ():
            f_one_third = f_one_third_calc(flux)
        if leaks_detected is () and flux is not ():
            leaks_detected = np.zeros(len(flux))
        try:
            length_in = len(x_pos)
            if not len(x_pos) == len(y_pos) == len(z_pos) == len(flux) == len(f_one_third):
                raise ValueError("x_pos, y_pos, z_pos, flux, f_one_third and leaks_detected must "
                                 "be equal length")
        except TypeError:
            length_in = 1

        if capacity == 0:
            self.x = np.array(x_pos)
            self.y = np.array(y_pos)
            self.z = np.array(z_pos)
            # print("typeflux", type(flux))
            # print("farr", type(np.asarray(flux)))
            # print("type selfflux", type(self.flux))
            self.flux = np.array(flux)
            self.f_one_third = np.array(f_one_third)
            self.leaks_detected = np.array(leaks_detected) if leaks_detected != () \
                else np.zeros(length_in)
        else:
            self.x = np.zeros(capacity)
            self.y = np.zeros(capacity)
            self.z = np.zeros(capacity)
            self.flux = np.zeros(capacity)
            self.f_one_third = np.zeros(capacity)
            self.leaks_detected = np.zeros(capacity)
            self.x[0:length_in] = x_pos
            self.y[0:length_in] = y_pos
            self.z[0:length_in] = z_pos
            self.flux[0:length_in] = flux
            self.f_one_third[0:length_in] = f_one_third
            self.leaks_detected[0:length_in] = np.array(leaks_detected) if leaks_detected != () \
                else np.zeros(length_in)
        self.n_leaks = length_in

    def __str__(self):
        msg = "Leak:: Count: {}  Mean: {}  Max: {}".format(len(self.flux), np.mean(self.flux),
                                                           np.max(self.flux))
        return msg

    def extend(self, leak_obj_in):
        """
        Add a new leak
        Inputs:
            leak_obj_in     a Leak object
        """
        if len(self.x) - leak_obj_in.n_leaks - self.n_leaks >= 0:
            # print("self.n_leaks", self.n_leaks)
            # print("len self.x", len(self.x))
            # print("len self.flux", len(self.flux))
            # print("leak_obj_in.n_leaks", leak_obj_in.n_leaks)
            # print("actual number of leak_obj_in x", len(leak_obj_in.x))
            # print("actual number of leak_obj_in flux", len(leak_obj_in.flux))
            if self.n_leaks < 0:
                return
            # print("extra space", len(self.x) - leak_obj_in.n_leaks - self.n_leaks)
            # print("num new leaks", leak_obj_in.n_leaks)
            new_n_leaks = self.n_leaks + leak_obj_in.n_leaks
            # print("new_n_leaks", new_n_leaks)
            # print("i don't even know what this is", self.x[self.n_leaks:new_n_leaks])
            # print("flux input array", self.flux[self.n_leaks:new_n_leaks])
            self.x[self.n_leaks:new_n_leaks] = leak_obj_in.x
            self.y[self.n_leaks:new_n_leaks] = leak_obj_in.y
            self.z[self.n_leaks:new_n_leaks] = leak_obj_in.z
            if len(self.flux) < self.n_leaks + leak_obj_in.n_leaks:
                np.append(self.flux, leak_obj_in.flux)
            else:
                self.flux[self.n_leaks:new_n_leaks] = leak_obj_in.flux
            self.f_one_third[self.n_leaks:new_n_leaks] = leak_obj_in.f_one_third
            self.leaks_detected[self.n_leaks:new_n_leaks] = leak_obj_in.leaks_detected
        else:
            # print("mystery clause is false")
            self.x = np.append(self.x, leak_obj_in.x)
            self.y = np.append(self.y, leak_obj_in.y)
            self.z = np.append(self.z, leak_obj_in.z)
            self.flux = np.append(self.flux, leak_obj_in.flux)
            self.f_one_third = np.append(self.f_one_third, leak_obj_in.f_one_third)
            self.leaks_detected = np.append(self.leaks_detected, leak_obj_in.leaks_detected)
        if leak_obj_in.n_leaks < 0:
            # print("warning less than zero")
            pass
        self.n_leaks += leak_obj_in.n_leaks

    def delete_leaks(self, indexes_to_delete):
        """
        Delete all parameters associated with leaks at indexes 'indexes_to_delete'
        Inputs:
            indexes_to_delete       A list of leak indexes to delete, or the string 'all'
        """
        if isinstance(indexes_to_delete, str):
            if indexes_to_delete == "all":
                indexes_to_delete = list(range(0, self.n_leaks))
            else:
                raise ValueError("indexes_to_delete must be a scalar, an array or the str 'all'")
        self.x = np.delete(self.x, indexes_to_delete)
        self.y = np.delete(self.y, indexes_to_delete)
        self.z = np.delete(self.z, indexes_to_delete)
        self.flux = np.delete(self.flux, indexes_to_delete)
        self.f_one_third = np.delete(self.f_one_third, indexes_to_delete)
        self.leaks_detected = np.delete(self.leaks_detected, indexes_to_delete)
        # Assert lengths are all the same
        assert len(self.x) == len(self.y)
        assert len(self.x) == len(self.z)
        assert len(self.x) == len(self.flux)
        assert len(self.x) == len(self.f_one_third)
        assert len(self.x) == len(self.leaks_detected)

        # TODO
        temp_previous = self.n_leaks
        try:
            self.n_leaks -= len(indexes_to_delete)
        except TypeError:
            # TODO I think this should be = 1
            self.n_leaks -= 1
        if self.n_leaks < 0:
            print("error")
            print(temp_previous)
            print(self.n_leaks)
            print(indexes_to_delete)
            raise AssertionError("n_leaks can't be negative")


def leak_objects_generator(dist_type, leak_data_path):
    """
    leak_objects is a parent function that will be called to initialize gas fields
    Inputs:
        dist_type           Type of leak distribution to be used
        leak_data_path      Path to a leak data file
    """
    leak_data_path = "InputData/DataObjectInstances/" + leak_data_path
    leak_data = pickle.load(open(leak_data_path, "rb"))

    # Number of leaking components at each well (Poisson distribution)
    detection_types, leaks_per_well = leak_data.leak_sizes.keys(), 0
    # This sums the leaks found per well by every method in leak_data
    for key in detection_types:
        leaks_per_well += len(leak_data.leak_sizes[key]) / leak_data.well_counts[key]
    return leak_data, leaks_per_well


def make_leaks(n_leaks, gas_field):
    """
    TODO
    """
    leak_params = gas_field.leak_params
    detection_methods = list(leak_params.leak_sizes.keys())
    leaks_per_well = init_leaks_per_well(detection_methods, leak_params)
    flux = []
    round_err = []
    counter = -1
    for method in detection_methods:
        counter += 1
        n_leaks_key = leaks_per_well[counter] / sum(leaks_per_well) * n_leaks
        # print(method, n_leaks_key)
        if gas_field.dist_type == "bootstrap":
            new_leaks = bootstrap_make_leak_sizes(n_leaks_key, method, leak_params)
        elif gas_field.dist_type == "power_law":
            new_leaks = power_law_make_leak_sizes(n_leaks_key, method, leak_params,
                                                  gas_field.power_law_stats)
        elif gas_field.dist_type == "lognormal":
            new_leaks = lognormal_make_leak_sizes(n_leaks_key, method, leak_params,
                                                  gas_field.lognormal_stats)
        else:
            msg = "Leak distribution {} unsupported in GasField, must be '{}' or '{}'"
            raise NameError(msg.format(gas_field.dist_type, "bootstrap", "power_law"))
        flux = np.concatenate((flux, new_leaks)) if len(flux) != 0 else new_leaks
        # print("flux len so far", len(flux))
        round_err.append(n_leaks_key % 1)
    # Add leaks omitted due to inability to add fractional leaks
    # The "round" function in the following line is intended to eliminate floating point errors.
    chooser = np.random.uniform(0, sum(round_err), round(sum(round_err)))
    # print("chooser", chooser)
    error_intervals = np.cumsum(round_err)
    for choose in chooser:
        ind = 0
        # Add a leak from the appropriate detection method
        while choose > error_intervals[ind]:
            ind += 1
        if gas_field.dist_type == "bootstrap":
            new_leak = bootstrap_make_leak_sizes(1, detection_methods[ind], leak_params)
        elif gas_field.dist_type == "power_law":
            new_leak = power_law_make_leak_sizes(1, detection_methods[ind], leak_params,
                                                 gas_field.power_law_stats)
        elif gas_field.dist_type == "lognormal":
            new_leak = lognormal_make_leak_sizes(1, detection_methods[ind], leak_params,
                                                 gas_field.lognormal_stats)
        flux = np.concatenate((flux, new_leak))
        # print("flux len so far, in rounderr step", len(flux))
    np.random.shuffle(flux)

    x = list(np.random.uniform(-gas_field.well_length / 2, gas_field.well_length / 2, n_leaks))
    y = list(np.random.uniform(-gas_field.well_length / 2, gas_field.well_length / 2, n_leaks))
    z = list(np.random.uniform(0, gas_field.h0_max, n_leaks))
    f_one_third = f_one_third_calc(flux)
    # print(len(x), len(y), len(z), len(f_one_third), len(flux))
    return Leak(x_pos=x, y_pos=y, z_pos=z, flux=flux,
                f_one_third=f_one_third, leaks_detected=[0] * len(x))


# TODO is it init or just set?
def init_leaks_per_well(detection_methods, leak_params):
    """
    Generate the appropriate number of leaks from the distribution associated with each detection
    method
    """
    leaks_per_well = []
    # Calculate leaks per well identified with each detection method stored in leak_params
    for method in detection_methods:
        n_leaks = len(leak_params.leak_sizes[method])
        # print("initied leaks")
        n_wells = leak_params.well_counts[method]
        leaks_per_well.append(n_leaks / n_wells)
    return leaks_per_well


def bootstrap_make_leak_sizes(n_leaks, method, leak_params):
    """
    Create leaks using a bootstrap method
    n_leaks                 number of leaks to generate
    gas_field               a GasField object
    """
    #print("type(method): {} type(n_leaks): {}".format(type(method), type(n_leaks)))
    return np.random.choice(leak_params.leak_sizes[method], int(n_leaks))


# power_law_min, alpha):
def power_law_make_leak_sizes(n_leaks, method, leak_params, power_law_stats):
    """
    Create leaks using a power law
    n_leaks                 number of leaks to generate
    gas_field               a GasField object
    """
    n_leaks = int(n_leaks)
    small_leaks = np.random.choice(leak_params.leak_sizes[method], n_leaks)
    # num_SOMETHING = len(leak_params.leak_sizes[method])
    # luck = np.random.rand(num_SOMETHING)
    # print(luck)
    # power_leak_count = np.sum(luck < power_fraction)
    power_leak_pos = small_leaks > power_law_stats["min"]
    power_leak_count = np.sum(power_leak_pos)
    # print("Power law n_leaks: {}".format(n_leaks))
    # print("Power law leaks above min: {}".format(power_leak_count))
    luck = np.random.rand(power_leak_count)
    power_leaks = power_law_stats["min"] * luck ** (1 / (1 - power_law_stats["alpha"]))
    overlimit = np.where(power_leaks > power_law_stats["max"])[0]
    len_pl_b4 = len(power_leaks)  # print
    power_leaks = np.delete(power_leaks, overlimit)
    len_pl_aftr = len(power_leaks)  # print
    power_leak_pos = np.delete(np.where(power_leak_pos)[0], overlimit)
    small_leaks = np.delete(small_leaks, power_leak_pos)
    # print("Power law number of overlimit: {}".format(len(overlimit)))
    # print("Power law small leaks after removal: {}".format(len(small_leaks)))
    # print("Power law large leaks: {}".format(len(power_leaks)))
    # print("Max size of small leaks: {}".format(np.max(small_leaks)))
    new_leaks = np.concatenate((small_leaks, power_leaks))
    if len(new_leaks) > 0:
        max_ = np.max(new_leaks)
        if max_ > 99:
            print("maximum", max_)
            print("pl max", power_law_stats["max"])
            print("num_overlimit", np.sum(overlimit))
            print(len_pl_b4, len_pl_aftr)
    assert len(new_leaks) == n_leaks
    return new_leaks


def lognormal_make_leak_sizes(n_leaks, method, leak_params, lognormal_stats):
    """
    Generate a hybrid set of leaks just like the power law, but use a lognormal.
    This will make sure that the expected number of leaks actually gets generated so that nothing
    breaks. Presumably the little leaks won't matter at all.
    :param n_leaks: number of leaks to generate
    :param method:
    :param leak_params: a structure that holds the Ft.Worth distribution?
    :param lognormal_stats: Dict with "min" (g/s), "mu" (kg/h), and "sigma" (kg/h). Well,
    the latter aren't ACTUALLY kg/h, they are in the units that when put in a lognormal give you
    kg/h.
    :return: list of leak fluxes in g/s
    """
    n_leaks = int(n_leaks)
    if n_leaks == 0:
        return []

    # generate the small leaks.
    small_leaks = np.random.choice(leak_params.leak_sizes[method], n_leaks)
    big_leak_threshold = np.percentile(leak_params.leak_sizes[method], lognormal_stats[
        "small_frac"])
    # print("Big-leak threshold: {} g/s".format(big_leak_threshold))

    # pull big leaks from lognormal
    lognormal_leak_pos = small_leaks >= big_leak_threshold
    lognormal_leak_count = np.sum(lognormal_leak_pos)
    # print("{} big leaks to make.".format(lognormal_leak_count))

    lognormal_flux = np.random.lognormal(np.log(lognormal_stats["mu"]),
                                         lognormal_stats["sigma"],
                                         lognormal_leak_count) * kgph_to_gps
    # unlike power law, we won't cut off this distribution, because it cuts itself off

    # Lognormal leaks are replacing, so get rid of the ones they replace
    # print("Small leaks before: {}  Removing: {}".format(len(small_leaks),
    #                                                     np.sum(lognormal_leak_pos)))
    small_leaks = small_leaks[small_leaks < big_leak_threshold]
    # print("Lognormal: Made {} small leaks, {} big leaks.".format(len(small_leaks),
    #                                                              lognormal_leak_count))
    # if len(small_leaks) > 0:
    #     print("Small leaks after: {}".format(len(small_leaks)))
    # else:
    #     print("No small leaks left!")

    new_leaks = np.concatenate((small_leaks, lognormal_flux))
    # print("New leaks: {}".format(len(new_leaks)))

    # print("n_leaks: {}  small_leaks: {} lognormal_flux: {}".format(n_leaks, len(small_leaks),
    #                                                               len(lognormal_flux)))
    # if len(small_leaks) > 0:
    #    print("Max size of small_leaks: {}".format(np.max(small_leaks)))
    assert len(new_leaks) == n_leaks

    return new_leaks


def f_one_third_calc(flux):
    """
    Computes the f_one_third leak dispersion parameter given a leak flux
    Inputs:
        flux        Leakage rate. Must be an iterable list or array [g/s]
    Return:
        f_one_third     Leak dispersion parameter
    """
    f_factor = g / np.pi * (1 / RHO_METHANE - 1 / RHO_AIR)  # Buoyancy flux [m^4/s^3]
    f_one_third = []
    for item in flux:
        f_one_third.append((f_factor * item) ** (1 / 3))
    return f_one_third
