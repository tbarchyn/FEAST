import os
import numpy as np
import pickle
from scipy.ndimage.filters import gaussian_filter
from kommons import KairosException
import multiprocessing
from multiprocessing import Pool
from GeneralClassesFunctions.plotting_functions import time_series

from feast_constants import *

# Plotting bits
import pylab
from matplotlib.font_manager import fontManager, FontProperties
from matplotlib.ticker import FuncFormatter


def load_realization(abspath, filename):
    file = os.path.join(abspath, filename)
    res = pickle.load(open(file, "rb"))
    return res


def emission_prevented_over_null(key, res):
    """
    Return the difference in emissions under the Null detection model and the tech model
    specificed by "key"
    :param key: The tech_dict key, e.g., "AD" or "MIR"
    :param res: The Results object
    :return: float
    """
    emission = np.sum(res.tech_dict["Null"].leakage) - np.sum(res.tech_dict[key].leakage)
    return emission


def emission_prevented_over_none(key, res):
    emission = np.sum(res.no_repair_leakage) - np.sum(res.tech_dict[key].leakage)
    return emission


def get_baseline_emission_over_null(key, abspath):
    """
    Return a dictionary with metadata and saved-emissions values, specifically for the baseline
    LDAR method. Similar to get_point_emission_over_null, except that additional metadata about
    the run is included as well.
    :param key: Should be "MIR"
    :param abspath: Directory containing baseline data, typically top_dir/ogi_baseline
    :return: dictionary
    """
    files = [f for f in os.listdir(abspath) if os.path.isfile(os.path.join(abspath, f))]
    emission_over_null = []
    for filename in files:
        res = load_realization(abspath, filename)
        emission_over_null.append(emission_prevented_over_null(key, res))

    survey_interval = res.tech_dict[key].survey_interval  # last one is as good as any

    result = {
        "survey_interval": survey_interval,
        "emission_over_null": emission_over_null,
        "delta_t": res.time.delta_t,
        "sim_duration": res.time.end_time,
        "site_count": res.gas_field.site_count,
        "gas_field": res.gas_field,
        "tech_dict": res.tech_dict,
        "time": res.time
    }
    return result


def emission_prevented_over_null_map(params):
    key = params["key"]
    inst_params_index = params["inst_params_index"]
    survey_interval_index = params["survey_interval_index"]
    res = load_realization(params["abspath"], params["filename"])
    if inst_params_index is None:
        inst_params_index = res.tech_dict[key].inst_params_index
        survey_interval_index = res.tech_dict[key].survey_interval_index
        inst_param = res.tech_dict[key].inst_params
        survey_interval = res.tech_dict[key].survey_interval
    else:
        if inst_params_index != res.tech_dict[key].inst_params_index or \
                        survey_interval_index != res.tech_dict[key].survey_interval_index:
            raise KairosException("Sim results mixed up!")

    emission_prevented_over_null(key, res)


def get_point_emission_over_null(key, abspath):
    """
    Given a path that points to a directory full of realizations of the same instrument
    parameters and survey intervals, computes the emissions saved vs. the null model in each
    realization, and returns a dictionary containing metadata and a list of all of the
    saved-emissions values.
    :param key: Should be "AD"
    :param abspath: Path to a set of realization results. The first one it picks up will be the
    gold standard; if any results files in the directory have different instrument params or
    survey intervals, it will barf.
    :return: dictionary with metadata and a list of saved-emissions values.
    """
    files = [f for f in os.listdir(abspath) if os.path.isfile(os.path.join(abspath, f))]

    emission_over_null = []
    true_leak_fluxes = []
    inst_params_index = None
    survey_interval_index = None
    inst_param = None
    survey_interval = None
    for filename in files:
        res = load_realization(abspath, filename)
        if inst_params_index is None:
            inst_params_index = res.tech_dict[key].inst_params_index
            survey_interval_index = res.tech_dict[key].survey_interval_index
            inst_param = res.tech_dict[key].inst_params
            survey_interval = res.tech_dict[key].survey_interval
        else:
            if inst_params_index != res.tech_dict[key].inst_params_index or \
                            survey_interval_index != res.tech_dict[key].survey_interval_index:
                raise KairosException("Sim results mixed up!")

        emission_over_null.append(emission_prevented_over_null(key, res))
        # print("appending leak fluxes")
        true_leak_fluxes.append(res.leak_list.flux)

    print("true_leak_fluxes length: {}".format(len(true_leak_fluxes)))
    result = {
        "inst_params_index": inst_params_index,
        "inst_params": inst_param,
        "survey_interval_index": survey_interval_index,
        "survey_interval": survey_interval,
        "true_leak_fluxes": true_leak_fluxes,
        "emission_over_null": emission_over_null
    }
    return result


def extract_emission_from_result_file(fullpath):
    abspath, filename = os.path.split(fullpath)
    try:
        res = load_realization(abspath, filename)
    except:
        print("Failed to load {}. Continuing.".format(fullpath))
        return []

    result = {}
    for tech in res.tech_dict:
        tech_res = {}
        if tech in TECH_NAME:
            tech_res = {
                "inst_params_index": res.tech_dict[tech].inst_params_index,
                "inst_params": res.tech_dict[tech].inst_params,
                "survey_interval_index": res.tech_dict[tech].survey_interval_index,
                "survey_interval": res.tech_dict[tech].survey_interval,
            }
        tech_res["emission_over_null"] = emission_prevented_over_null(tech, res)
        tech_res["site_count"] = res.gas_field.site_count
        tech_res["end_time"] = res.time.end_time

        result[tech] = tech_res

    return result


def plot_all_timeseries(top_dir):
    filelist = []
    dirs = [os.path.join(os.path.abspath(top_dir), folder) for folder in os.listdir(top_dir)]
    for d in dirs:
        files = [os.path.join(os.path.abspath(top_dir), d, f) for f in os.listdir(d)]
        filelist.extend(files)

    for file in filelist:
        print("{}".format(file))
        time_series(file)


def sum_combined_baseline_runs(top_dir):
    # Sum up all the runs by inst_param/survey_interval directory
    # loop over parameter index folders, pull all the filenames,
    # pool extract_emission_from_result_file
    p = Pool()
    print("Collect multiprocessing with {} nodes.".format(multiprocessing.cpu_count()))

    emission_summary_list = []
    filelist = []
    dirs = [os.path.join(os.path.abspath(top_dir), folder) for folder in os.listdir(top_dir)]
    for d in dirs:
        try:
            print("Summing data for {}".format(d))

            files = [os.path.join(os.path.abspath(top_dir), d, f) for f in os.listdir(d)]
            print("  Found {} realization files.".format(len(files)))

            # pull out emission over null just for this dir (AD inst_params/survey interval)
            emission_dict_list = p.map(extract_emission_from_result_file, files)

            total_ogi = np.sum(np.array([ed[BASELINE_NAME]["emission_over_null"] for ed in
                                         emission_dict_list]))
            total_ogi_sites = np.sum(
                np.array([ed[BASELINE_NAME]["site_count"] for ed in emission_dict_list]))
            total_ogi_days = np.mean(
                np.array([ed[BASELINE_NAME]["end_time"] for ed in emission_dict_list]))

            total_ls = np.sum(np.array([ed[TECH_NAME]["emission_over_null"] for ed in
                                        emission_dict_list]))
            total_ls_sites = np.sum(
                np.array([ed[TECH_NAME]["site_count"] for ed in emission_dict_list]))
            total_ls_days = np.mean(
                np.array([ed[TECH_NAME]["end_time"] for ed in emission_dict_list]))

            em_summary = {
                "baseline_over_null": total_ogi,
                "baseline_total_sites": total_ogi_sites,
                "baseline_total_days": total_ogi_days,
                "tech_over_null": total_ls,
                "tech_total_sites": total_ls_sites,
                "tech_total_days": total_ls_days,
                "inst_params_index": emission_dict_list[0][TECH_NAME][
                    "inst_params_index"],
                "survey_interval_index": emission_dict_list[0][TECH_NAME][
                    "survey_interval_index"],
                "inst_params": emission_dict_list[0][TECH_NAME]["inst_params"],
                "survey_interval": emission_dict_list[0][TECH_NAME][
                    "survey_interval"]
            }
            emission_summary_list.append(em_summary)

        except Exception as e:
            print("Couldn't pull any files from {}: {}".format(d, e))

    return emission_summary_list


def sum_tiny_runs(top_dir, tech=TECH_NAME, force_rebuild=False):
    # aggregate lots of TinyResults into a list that can be given to plot_contour_combined_sum

    emission_summary_list = []
    filelist = []
    summary_filename = os.path.abspath(top_dir) + "_summary_list.p"
    if os.path.isfile(summary_filename) and not force_rebuild:
        print("Read summary list from summary file.")
        edl = pickle.load(open(summary_filename, "rb"))
        return edl

    dirs = [os.path.join(os.path.abspath(top_dir), folder) for folder in os.listdir(top_dir)]
    for d in dirs:
        try:
            print("Summing data for {}".format(d))

            files = [os.path.join(os.path.abspath(top_dir), d, f) for f in os.listdir(d)]
            print("  Found {} realization files.".format(len(files)))

            ogi_emission = []
            null_emission = []
            ls_emission = []
            extreme_leaks = []
            total_sites = []
            max_leaks = []
            total_leaks = []

            for file in files:
                tr = pickle.load(open(file, "rb"))
                ogi_emission.append(tr.total_leakage["MIR"])  # omg this is fragile
                null_emission.append(tr.total_leakage["Null"])
                ls_emission.append(tr.total_leakage[tech])
                total_sites.append(tr.site_count)
                extreme_leaks.append(tr.extreme_leaks)
                max_leaks.append(tr.max_leak)
                total_leaks.append(tr.total_leaks) #summing over whatever random order you picked
                #  up the files in.

                em_summary = {
                    "gas_field_type": tr.gas_field_type,
                    "baseline_over_null": np.sum(null_emission) - np.sum(ogi_emission),
                    "baseline_total_sites": np.sum(total_sites),
                    "baseline_total_days": tr.time.end_time,
                    "baseline_survey_interval": tr.tech_params[BASELINE_NAME]["survey_interval"],
                    "baseline_distance": tr.tech_params[BASELINE_NAME]["distance"],
                    "tech_over_null": np.sum(null_emission) - np.sum(ls_emission),
                    "tech_total_sites": np.sum(total_sites),
                    "tech_total_days": tr.time.end_time,
                    "inst_params_index": tr.tech_params[tech]["inst_params_index"],
                    "survey_interval_index": tr.tech_params[tech]["survey_interval_index"],
                    "inst_params": tr.tech_params[tech]["inst_params"],
                    "survey_interval": tr.tech_params[tech]["survey_interval"],
                    "max_leak": np.max(max_leaks),
                    "leak_extreme_sizes": tr.leak_extreme_sizes,
                    "extreme_leaks": np.sum(np.array(extreme_leaks), 0),
                    "total_leaks": np.sum(np.array(total_leaks))
                }
                emission_summary_list.append(em_summary)

        except Exception as e:
            print("Couldn't pull any files from {}: {}".format(d, e))

        pickle.dump(emission_summary_list, open(summary_filename, "wb"))
    return emission_summary_list


def hist_emission_dict(plt, emission_dict_list):
    gs_to_kg = (emission_dict_list[0]["delta_t"] * 86400 / 1000) / emission_dict_list[0][
        "site_count"] / (emission_dict_list[0]["sim_duration"] / 365)
    for emission_dict in emission_dict_list[1:]:
        plt.figure()
        bins = np.max([10, int(len(emission_dict["emission_over_null"]) / 10)])
        plt.hist(np.array(emission_dict["emission_over_null"]) * gs_to_kg, bins=bins)
        plt.title("Emissions Prevented for\ninst_param {:.1f} survey_interval {}".format(
            emission_dict["inst_params"]["center"] / mcfpd_to_gps,
            emission_dict["survey_interval"]))
        plt.xlabel("kg/year/site")
        print("Mean: {}  Std: {}".format(np.mean(np.array(emission_dict["emission_over_null"]) *
                                                 gs_to_kg), np.std(
            np.array(emission_dict["emission_over_null"]) * gs_to_kg)))


def generate_true_leak_stats(plt, emission_dict_list, threshold=20):
    true_leak_list = []
    for emission_dict in emission_dict_list[1:]:
        # In each dict for each realization, there is a list of lists of fluxes. Pull them all out.
        realization_truth = [leak for leak_list in emission_dict["true_leak_fluxes"]
                             for leak in leak_list]
        true_leak_list.extend(realization_truth)

    true_leak_list = np.array(true_leak_list)
    msg = "Ratio of leaks over {} Mscf/day to under: {}"
    print(msg.format(threshold,
                     np.sum(true_leak_list > threshold * mcfpd_to_gps) / len(true_leak_list)))
    msg = "Fraction of Leaks over {} Mscf/day per site per day: {}"
    print(msg.format(threshold, np.sum(true_leak_list > threshold * mcfpd_to_gps) / (
        emission_dict_list[0]["site_count"] *
        emission_dict_list[0]["sim_duration"])))

    plt.hist(true_leak_list / mcfpd_to_gps, bins=100);
    plt.gca().set_yscale("log")
    plt.xlabel("Leak size (Mscf/day)")


def report_combined_sum(plt, emission_summary_list):
    emi = emission_summary_list[0]
    print("FEAST Run Report")
    print("  Gas field type: {}".format(emi["gas_field_type"]))
    print("  Baseline OGI: Survey interval: {} days from {} m".format(
        emi["baseline_survey_interval"],
        emi["baseline_distance"]))
    print("---------")
    for emi in emission_summary_list:
        print(" Survey Interval {}, Inst. Params {}:".format(emi["survey_interval_index"],
                                                             emi["inst_params_index"]))
        if "extreme_leaks" in emi:
            msg = " Total leaks: {} Max leak: {:.1f} Mscf/day  Leaks ".format(
                emi["total_leaks"],
                emi["max_leak"] / mcfpd_to_gps)
        for level, count in zip(emi["leak_extreme_sizes"], emi["extreme_leaks"]):
            msg += " >{:.1f}: {}  ".format(level / mcfpd_to_gps, count)
        print(msg)

        print(
            "     Baseline sites: {}  Days: {} Total count (g/s): {:.0f}  kg/site/day: {:.0f}".format(
                emi["baseline_total_sites"],
                emi["baseline_total_days"],
                emi["baseline_over_null"],
                emi["baseline_over_null"] * (86400 / 1000) / emi["baseline_total_days"] / emi[
                    "baseline_total_sites"]))
        print("     Tech sites: {}  Days: {} Total count (g/s): {:.0f}  kg/site/day: {:.0f}".format(
            emi["tech_total_sites"],
            emi["tech_total_days"],
            emi["tech_over_null"],
            emi["tech_over_null"] * (86400 / 1000) / emi["tech_total_days"] / emi[
                "tech_total_sites"]))


def plot_contour_combined_sum(plt, emission_summary_list):
    # Contour-plot the results of sum_combined_baseline_runs

    #    gs_to_kg = (emission_dict_list[0]["delta_t"] * 86400 / 1000) / emission_dict_list[0][
    #        "site_count"] / (emission_dict_list[0]["sim_duration"] / 365)

    plt.rc('font', family='Carlito')
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    num_inst = 0
    num_surv = 0
    inst_val = set()
    surv_val = set()
    for emi in emission_summary_list:
        if emi["inst_params_index"] > num_inst:
            num_inst = emi["inst_params_index"]
        if emi["survey_interval_index"] > num_surv:
            num_surv = emi["survey_interval_index"]
        inst_val.add(emi["inst_params"]["center"] / mcfpd_to_gps)
        surv_val.add(emi["survey_interval"])

    inst_val = np.array(sorted(list(inst_val)))
    surv_val = np.array(sorted(list(surv_val)))
    print("Sensitivity values: {}".format(inst_val))
    print("Survey Interval values: {}".format(surv_val))

    summary_list = []  # 3xlen(emission_dict_list)
    print("WARNING: ASSUMES EMISSION is days*(g/s), and has been corrected for differences in "
          "delta_t. "
          "This might even be true.")
    for emi in emission_summary_list:
        tech_emission_prevented = emi["tech_over_null"] * (86400 / 1000) / emi[
            "tech_total_days"] / emi["tech_total_sites"]
        baseline_emission_prevented = emi["baseline_over_null"] * (86400 / 1000) / emi[
            "baseline_total_days"] / emi["baseline_total_sites"]
        summary_list.append([emi["inst_params_index"],
                             emi["survey_interval_index"],
                             tech_emission_prevented,
                             baseline_emission_prevented])
    summary_list = np.array(summary_list)

    tech_data = np.zeros((num_inst + 1, num_surv + 1))  # zero-based
    baseline_data = np.zeros((num_inst + 1, num_surv + 1))  # zero-based
    cont_data = np.zeros((num_inst + 1, num_surv + 1))  # zero-based

    for inst_idx in range(len(inst_val)):
        for surv_idx in range(len(surv_val)):
            inst_mask = summary_list[:, 0] == inst_idx
            surv_mask = summary_list[:, 1] == surv_idx
            val = np.mean(summary_list[np.logical_and(inst_mask, surv_mask), 2])
            tech_data[inst_idx][surv_idx] = val
            val = np.mean(summary_list[np.logical_and(inst_mask, surv_mask), 3])
            baseline_data[inst_idx][surv_idx] = val

    sigma = 1.0
    cont_data = gaussian_filter(tech_data / baseline_data, sigma)

    levels = np.array([0.5, 0.65, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.25])
    CS = plt.contour(surv_val, inst_val, cont_data, levels=levels);
    fmt = {}
    strs = []
    for lev in levels:
        strs.append("{:.0f}%".format(lev * 100))
    for l, s in zip(CS.levels, strs):
        fmt[l] = s

    plt.xlabel("Survey Interval (days)", fontsize=10, fontweight='bold')
    plt.ylabel("Instrument Threshold (Mscf/day)", fontsize=10, fontweight='bold')
    plt.clabel(CS, CS.levels, fmt=fmt)
    # plt.clabel(CS, inline=1, fontsize=10)
    plt.title("Comparative Emission Reduction\nOGI Baseline: {}-Day Interval".format(
        emission_summary_list[0]["baseline_survey_interval"]),
        fontsize=12,
        fontweight='bold')
    plt.savefig("EmissionReductionContour_{}m_{}days.png".format(
        emission_summary_list[0]["baseline_distance"],
        emission_summary_list[0]["baseline_survey_interval"]), dpi=300)

    plt.figure()
    CS2 = plt.contour(surv_val, inst_val, tech_data);
    plt.xlabel("Survey Interval (days)", fontsize=10, fontweight='bold')
    plt.ylabel("Instrument Threshold (Mscf/day)", fontsize=10, fontweight='bold')
    # plt.clabel(CS, CS.levels, fmt=fmt)
    plt.clabel(CS2, inline=1, fontsize=10)
    plt.title("Emission Reduction of Tech (kg/site/day)", fontsize=12, fontweight='bold')

    plt.figure()
    CS3 = plt.contour(surv_val, inst_val, baseline_data);
    plt.xlabel("Survey Interval (days)", fontsize=10, fontweight='bold')
    plt.ylabel("Instrument Threshold (Mscf/day)", fontsize=10, fontweight='bold')
    # plt.clabel(CS, CS.levels, fmt=fmt)
    plt.clabel(CS3, inline=1, fontsize=10)
    plt.title("Emission Reduction of Baseline (kg/site/day)", fontsize=12, fontweight='bold')

    return summary_list
