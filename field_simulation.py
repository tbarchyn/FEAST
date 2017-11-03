import copy
import numpy as np
import os
import multiprocessing
import pickle
from multiprocessing import Pool

from GeneralClassesFunctions import simulation_classes
# DetectionModules is used within an eval() function call.
import DetectionModules as Dm
from GeneralClassesFunctions.simulation_functions import new_leak_count, save_results
from GeneralClassesFunctions.leak_class_functions import make_leaks
from GeneralClassesFunctions.simulation_classes import GasField

from feast_constants import *

# Note: this makes it hard to open-source
import kommons
import logging

DETECTION_CLASSES = {
    "FID": Dm.fid.FID, "Null": Dm.null.Null, "AIR": Dm.ir.AIR, "MIR": Dm.ir.MIR,
    "DD": Dm.dd.DD, "AD": Dm.airborne_detector.AD, "HY": Dm.hybrid.HY
}


def compute_ogi_baseline(n_realizations=100,
                         gas_field_name="power_law",
                         site_count=100,
                         master_out_dir="Results"):
    """
    Just a function to run a big pile of OGIs for a baseline, and do it in parallel.
    :param n_realizations:
    :param gas_field_name:
    :param master_out_dir:
    :return:
    """

    print("OGI Multiprocessing on {} nodes".format(multiprocessing.cpu_count()))
    p = Pool()

    kwargs_list = []

    # Create the OGI control.

    out_dir = os.path.join(master_out_dir, OGI_DIR_NAME)
    for realization in range(n_realizations):
        gas_field = GasField(dist_type=gas_field_name,
                             null_repair_rate=null_repair_rate_per_day,
                             site_count=site_count)
        simulation_time = simulation_classes.Time()
        ogi_tech = Dm.ir.MIR(simulation_time, gas_field)  # use default survey_interval of 100 days
        null_tech = Dm.null.Null(simulation_time, gas_field)
        kwargs = {
            "gas_field": gas_field,
            "time": simulation_time,
            "tech_dict": {
                "MIR": ogi_tech,
                "Null": null_tech
            },
            "detection_techs": ["MIR"],
            "label": "OGI Baseline",
            "dir_out": out_dir,
        }
        kwargs_list.append(kwargs)

    p.map(field_simulation_map, kwargs_list)


def compute_contour_data(instrument_param_list, survey_interval_list=[7],
                         n_realizations=100,
                         gas_field_name="power_law",
                         ls_detection_model="legacy",
                         site_count=100,
                         baseline_survey_interval=100,
                         baseline_distance=10,
                         master_out_dir="Results",
                         max_leak=100,
                         fullResults=False):
    """
    Run all the sims necessary to fill in a contour plot with axes "threshold" and "Survey interval"
    :param instrument_param_list: a list of inst_param dicts. There may be more than one
    instrument parameter in each dict; we'll use "center" as the one that we plot against. This
    will require additional generalization if you want to add more axes of variation.

    This will generate a series of directories inside master_out_dir, each containing
    n_realizations Results objects, saved in a separate file. Note that this can be a lot of data.

    The first directory contains runs on the OGI control.

    :param survey_interval_list: A list of survey intervals in days
    :param n_realizations: Number of realizations to run for each combo.
    :param gas_field_name: The name of one of the gas_field distribution types, defined in GasField
    :param ls_detection_model: The name of a Leak Surveyor detection model, defined in AD
    :param master_out_dir: Where to stash all the results files.
    :return:
    """

    # Now run for all the AD configurations.
    kwargs_list = []
    for realization in range(n_realizations):
        for inst_idx, instrument_params in enumerate(instrument_param_list):
            for freq_idx, frequency in enumerate(survey_interval_list):
                # make a new simulation time every time
                simulation_time = simulation_classes.Time()
                gas_field = GasField(dist_type=gas_field_name,
                                     null_repair_rate=null_repair_rate_per_day,
                                     site_count=site_count,
                                     max_leak=max_leak)
                null_tech = Dm.null.Null(simulation_time, gas_field)
                ogi_tech = Dm.ir.MIR(simulation_time,
                                     gas_field,
                                     distance=baseline_distance,
                                     survey_interval=baseline_survey_interval)

                extra_arg = {
                    "detection_model_name": ls_detection_model,
                    "inst_params": instrument_params,
                    "survey_interval": frequency,
                    "inst_params_index": inst_idx,
                    "survey_interval_index": freq_idx
                }
                myAD = Dm.airborne_detector.AD(simulation_time, gas_field, **extra_arg)

                name = ls_detection_model + "_"
                name += "inst" + str(inst_idx) + "_"
                name += "freq" + str(freq_idx)
                # for key in instrument_params:
                #     name += key + str(instrument_params[key]) + "_"
                # name += "freq" + str(frequency)
                out_dir = os.path.join(master_out_dir, name)
                try:
                    os.mkdir(out_dir)
                except OSError:
                    pass
                    # print("Directory already exists: {}", out_dir)
                # print("{}".format(name))
                kwargs = {
                    "gas_field": gas_field,
                    "time": simulation_time,
                    "tech_dict": {
                        "AD": myAD,
                        "MIR": ogi_tech,
                        "Null": null_tech
                    },
                    "detection_techs": ["AD, MIR"],
                    "label": name,
                    "dir_out": out_dir,
                    "fullResults": fullResults
                }  # TODO: Set runID explicitly

                kwargs_list.append(kwargs)
                # kwargs_list.extend(np.repeat(kwargs, n_realizations))

    print("Multiprocessing on {} nodes".format(multiprocessing.cpu_count()))
    p = Pool()

    p.map(field_simulation_map, kwargs_list)
    # for kwargs in kwargs_list:
    # field_simulation(**kwargs)


def compute_hybrid_contour_data(instrument_param_list, survey_interval_list=[7],
                                n_realizations=100,
                                gas_field_name="power_law",
                                ls_detection_model="legacy",
                                site_count=100,
                                hybrid_ogi_survey_interval=720,
                                hybrid_ogi_distance=10,
                                baseline_survey_interval=100,
                                baseline_distance=10,
                                master_out_dir="Results",
                                max_leak=100,
                                fullResults=False):
    """
    Run all the sims necessary to fill in a contour plot with axes "threshold" and "Survey interval"
    :param instrument_param_list: a list of inst_param dicts. There may be more than one
    instrument parameter in each dict; we'll use "center" as the one that we plot against. This
    will require additional generalization if you want to add more axes of variation.

    This will generate a series of directories inside master_out_dir, each containing
    n_realizations Results objects, saved in a separate file. Note that this can be a lot of data.

    The first directory contains runs on the OGI control.

    :param survey_interval_list: A list of survey intervals in days
    :param n_realizations: Number of realizations to run for each combo.
    :param gas_field_name: The name of one of the gas_field distribution types, defined in GasField
    :param ls_detection_model: The name of a Leak Surveyor detection model, defined in AD
    :param master_out_dir: Where to stash all the results files.
    :return:
    """

    # Now run for all the AD configurations.
    kwargs_list = []
    for realization in range(n_realizations):
        for inst_idx, instrument_params in enumerate(instrument_param_list):
            for freq_idx, frequency in enumerate(survey_interval_list):
                # make a new simulation time every time
                simulation_time = simulation_classes.Time()
                gas_field = GasField(dist_type=gas_field_name,
                                     null_repair_rate=null_repair_rate_per_day,
                                     site_count=site_count,
                                     max_leak=max_leak)
                null_tech = Dm.null.Null(simulation_time, gas_field)
                ogi_tech = Dm.ir.MIR(simulation_time,
                                     gas_field,
                                     distance=baseline_distance,
                                     survey_interval=baseline_survey_interval)

                extra_arg = {
                    "detection_model_name": ls_detection_model,
                    "inst_params": instrument_params,
                    "ls_survey_interval": frequency,
                    "inst_params_index": inst_idx,
                    "survey_interval_index": freq_idx,
                    "ogi_survey_interval": hybrid_ogi_survey_interval,
                    "ogi_distance": hybrid_ogi_distance
                }
                myhyb = Dm.hybrid.HY(simulation_time, gas_field, **extra_arg)

                name = ls_detection_model + "_"
                name += "inst" + str(inst_idx) + "_"
                name += "freq" + str(freq_idx)
                # for key in instrument_params:
                #     name += key + str(instrument_params[key]) + "_"
                # name += "freq" + str(frequency)
                out_dir = os.path.join(master_out_dir, name)
                try:
                    os.mkdir(out_dir)
                except OSError:
                    pass
                    # print("Directory already exists: {}", out_dir)
                # print("{}".format(name))
                kwargs = {
                    "gas_field": gas_field,
                    "time": simulation_time,
                    "tech_dict": {
                        "HYB": myhyb,
                        "MIR": ogi_tech,
                        "Null": null_tech
                    },
                    "detection_techs": ["HY, MIR"],
                    "label": name,
                    "dir_out": out_dir,
                    "fullResults": fullResults
                }  # TODO: Set runID explicitly

                kwargs_list.append(kwargs)
                # kwargs_list.extend(np.repeat(kwargs, n_realizations))

    print("Multiprocessing on {} nodes".format(multiprocessing.cpu_count()))
    p = Pool()

    p.map(field_simulation_map, kwargs_list)
    # for kwargs in kwargs_list:
    # field_simulation(**kwargs)


def field_simulation_map(arg_dict):
    return field_simulation(**arg_dict)


def field_simulation(gas_field=None, atm=None, input_leaks=None, dir_out="Results", time=None,
                     econ_set=None, tech_dict=None, detection_techs=None, display_status=True,
                     label=None, runID=None, fullResults=False):
    """
    field_simulation generates a single realization of scenario. The scenario is defined by the
    input values.
    gas_field           a GasField object
    atm                 an Atmosphere object
    input_leaks         a list of leaks to be generated at each timestep
    dir_out             directory name in which to save results
    time                a Time object
    econ_set            a FinanceSettings object
    tech_dict           a dict of detection technology objects
    detection_techs     a list of detection technology identifying strings
    label               a string to include in the filenames of the realization results
    runID               a number to include in the filenames of the realization results
    """

    kommons.logging.do_default_config(level=logging.WARNING)

    # -------------- Define settings --------------
    # time defines parameters related to time in the model. Time units are days.
    if time is None:
        time = simulation_classes.Time()

    if gas_field is None:
        gas_field = simulation_classes.GasField()
    leak_list = copy.deepcopy(gas_field.initial_leaks)

    # Note: econ_settings are not used during the simulation, but are saved for use in post
    # simulation data processing
    if econ_set is None:
        econ_set = simulation_classes.FinanceSettings()

    if atm is None:
        atm = simulation_classes.Atmosphere(time.n_timesteps)

    if detection_techs is None:
        detection_techs = ["FID", "Null", "AIR", "MIR", "DD", "AD"]
    else:
        print("Running {}".format(detection_techs))

    new_leaks, no_repair_leakage = [], []
    if tech_dict is None:
        tech_dict = dict()
        for tech in detection_techs:
            tech_dict[tech] = DETECTION_CLASSES[tech](time, gas_field)
    elif "Null" not in tech_dict:
        print("Warning: Null tech not in dict. Adding it.")
        tech_dict["Null"] = Dm.null.Null(time, gas_field)

    # --- Establish base conditions
    # print("field_simulation run {}".format(runID))
    # print("field_simulation: Timesteps: {}. Current time: {} Time index: {}".format(
    #     time.n_timesteps,
    #     time.current_time,
    #     time.time_index))

    # -------------- Run the simulation --------------
    n_leaks = []
    for time.time_index in range(0, time.n_timesteps):
        time.current_time += time.delta_t
        # if display_status and time.current_time % int(time.end_time / 10) < time.delta_t:
        #     print("Currently evaluating time step " + str(time.time_index)
        #           + " of " + str(time.n_timesteps))
        if input_leaks is None:  # and LEAKSTHERE_ALREADY_WHY NOT THERE:
            new_leaks.append(make_leaks(new_leak_count(time, gas_field), gas_field))
            # print("New leaks on Day {:.1f}: {}".format(time.current_time,
            #                                            len(new_leaks[-1].flux)))
        else:
            new_leaks.append(input_leaks[time.time_index])
        no_repair_leakage.append(sum(leak_list.flux))
        leak_list.extend(new_leaks[-1])
        # Loop through each LDAR program:
        for tech_obj in tech_dict.values():
            # the following is a terrible idea, it should be 2-D
            # just append the number of leaks for this tech for this step in a long list.
            n_leaks.append(tech_obj.leaks.n_leaks)
            # append leakage in this step (not necessarily total daily leakage) for this tech
            tech_obj.leakage.append(sum(tech_obj.leaks.flux))
            # Find and remove leaks
            tech_obj.detection(time, gas_field, atm)
            # Tack on the new leaks for this time step
            tech_obj.leaks.extend(new_leaks[-1])
    # -------------- Save results --------------
    results = simulation_classes.Results(time, gas_field, tech_dict, leak_list, no_repair_leakage,
                                         atm, econ_set, new_leaks, n_leaks)
    save_results(dir_out, results, label=label, runID=runID, fullResults=fullResults)
