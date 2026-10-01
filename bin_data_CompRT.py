import pandas as pd
import numpy as np
import pyopenms as oms
import os
import argparse
import time

from itertools import groupby


def args_setting():
    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-meta', type=str, dest='meta_filepath', default= r"F:/extra_mzml/meta.xlsx")
    parser.add_argument('-data', type=str, dest='raw_folderpaths', default= r"F:/extra_mzml")
    parser.add_argument('-save', dest='save_path', default= '../npy data')
    parser.add_argument('-mode', type=str, dest='mode', default='default')

    # GCxGC-specific
    parser.add_argument('-mod', type=float, dest='modulation_period', default=4.0)
    parser.add_argument('-rt1_bin', type=float, dest='rt1_bin_width', default=2.0)
    parser.add_argument('-rt2_bin', type=float, dest='rt2_bin_width', default=0.01)

    # x_comp grid
    parser.add_argument('-xcomp_start', type=float, dest='xcomp_start', default=500.0)
    parser.add_argument('-xcomp_end', type=float, dest='xcomp_end', default=1468.0)

    args = parser.parse_args()
    return args


def decompose_gcxcg_rt(rt_total, modulation_period):
    """
    Convert total RT into GCxGC-style components.
    rt2 = RT within one modulation period
    rt1 = RT excluding rt2
    """
    rt2 = rt_total % modulation_period
    rt1 = rt_total - rt2
    return rt1, rt2


def get_binned_xcomp(rt_total, modulation_period, rt1_bin_width, rt2_bin_width):
    """
    Decompose RT, bin RT1 and RT2 first, then compute x_comp.
    """
    rt1, rt2 = decompose_gcxcg_rt(rt_total, modulation_period)

    rt1_binned = round(round(rt1 / rt1_bin_width) * rt1_bin_width, 6)
    rt2_binned = round(round(rt2 / rt2_bin_width) * rt2_bin_width, 6)

    x_comp = rt1_binned + (rt2_binned / modulation_period)
    return rt1_binned, rt2_binned, x_comp


def build_xcomp_axis(xcomp_start, xcomp_end, n_rows):
    """
    Build x_comp axis as bin centers.
    """
    edges = np.linspace(xcomp_start, xcomp_end, n_rows + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers


def main():
    args = args_setting()
    meta_filepath = args.meta_filepath
    folder = args.raw_folderpaths
    save_path = args.save_path
    mode = args.mode
    modulation_period = args.modulation_period
    rt1_bin_width = args.rt1_bin_width
    rt2_bin_width = args.rt2_bin_width
    xcomp_start = args.xcomp_start
    xcomp_end = args.xcomp_end

    data = pd.read_excel(meta_filepath, index_col=0, engine="openpyxl")
    data.index = data.index.map(str)

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    filepaths = os.listdir(folder)

    if save_path is not None:
        processed_files = os.listdir(save_path)
        processed_files = [file.split(".")[0] for file in processed_files if file.endswith(".npy")]
        filepaths = [
            file for file in filepaths
            if (file.split(".")[0] not in processed_files) and (file.split(".")[0] in data.index)
        ]

    for file in filepaths:
        if file.endswith(".mzML"):
            sample = file.split(".")[0]

            if sample in data.index.tolist():
                filepath = os.path.join(folder, file)
                exp = oms.MSExperiment()

                try:
                    oms.MzMLFile().load(filepath, exp)
                    if mode == "adaptive":
                        __adaptive_binning(
                            exp,
                            save_path=save_path,
                            filetype="mzML",
                            modulation_period=modulation_period,
                            rt1_bin_width=rt1_bin_width,
                            rt2_bin_width=rt2_bin_width,
                            xcomp_start=xcomp_start,
                            xcomp_end=xcomp_end
                        )
                    else:
                        __default_binning(
                            exp,
                            save_path=save_path,
                            filetype="mzML",
                            modulation_period=modulation_period,
                            rt1_bin_width=rt1_bin_width,
                            rt2_bin_width=rt2_bin_width,
                            xcomp_start=xcomp_start,
                            xcomp_end=xcomp_end
                        )
                except Exception as e:
                    print(f"ERROR: File {file} could not be loaded! {e}")

        elif file.endswith(".mzdata.xml"):
            sample = file.split('.')[0]

            if sample in data.index.tolist():
                filepath = os.path.join(folder, file)
                exp = oms.MSExperiment()

                try:
                    oms.MzDataFile().load(filepath, exp)
                    if mode == "adaptive":
                        __adaptive_binning(
                            exp,
                            save_path=save_path,
                            filetype="mzdata",
                            modulation_period=modulation_period,
                            rt1_bin_width=rt1_bin_width,
                            rt2_bin_width=rt2_bin_width,
                            xcomp_start=xcomp_start,
                            xcomp_end=xcomp_end
                        )
                    else:
                        __default_binning(
                            exp,
                            save_path=save_path,
                            filetype="mzdata",
                            modulation_period=modulation_period,
                            rt1_bin_width=rt1_bin_width,
                            rt2_bin_width=rt2_bin_width,
                            xcomp_start=xcomp_start,
                            xcomp_end=xcomp_end
                        )
                except Exception as e:
                    print(f"ERROR: File {file} could not be loaded! {e}")


def __adaptive_binning(
    raw_LCMS,
    dim=(856, 1024),
    xcomp_start=500.0,
    xcomp_end=1468.0,
    mz_start=40,
    mz_end=800,
    modulation_period=4.0,
    rt1_bin_width=2.0,
    rt2_bin_width=0.01,
    save_path=None,
    filename=None,
    filetype="mzML"
):
    # rows = x_comp bins
    xcomp_bins = pd.interval_range(xcomp_start, xcomp_end, periods=dim[0])
    xcomp_axis = build_xcomp_axis(xcomp_start, xcomp_end, dim[0])

    # adaptive m/z bins
    mz_bins = []
    mz_bins_densities = [0.012, 0.048, 0.1, 0.128, 0.136, 0.099, 0.05, 0.049, 0.048, 0.065, 0.026, 0.009, 0.006, 0.045, 0.111, 0.063, 0.001, 0.001, 0.004]

    for i in range(0, int((mz_end - mz_start) / 50)):
        if i == 0:
            mz_bins.extend(
                np.linspace(
                    mz_start,
                    mz_start + 50,
                    int(np.round((dim[1] + len(mz_bins_densities) - 2) * mz_bins_densities[i], 0))
                )
            )
        else:
            mz_bins.extend(
                np.linspace(
                    mz_bins[-1],
                    mz_bins[-1] + 50,
                    int(np.round((dim[1] + len(mz_bins_densities) - 2) * mz_bins_densities[i], 0))
                )[1:]
            )

    if len(mz_bins) != dim[1]:
        mz_bins = mz_bins[:dim[1]]

    mz_axis = np.array(mz_bins, dtype=float)

    binned_matrix = np.zeros((dim[0], dim[1]))

    print("\n=== X_COMP DIAGNOSTIC ===")
    all_xcomp = []

    for i in range(raw_LCMS.getNrSpectra()):
        if raw_LCMS[i].getMSLevel() == 1:
            rt_total = raw_LCMS[i].getRT()

            _, _, x_comp = get_binned_xcomp(
                rt_total=rt_total,
                modulation_period=modulation_period,
                rt1_bin_width=rt1_bin_width,
                rt2_bin_width=rt2_bin_width
            )

            if xcomp_start < x_comp < xcomp_end:
                all_xcomp.append(x_comp)

    unique_xcomp = sorted(set(all_xcomp))
    print("Total MS1 scans in range:", len(all_xcomp))
    print("Unique x_comp positions:", len(unique_xcomp))
    print("First 10 x_comp:", unique_xcomp[:10])
    print("Last 10 x_comp:", unique_xcomp[-10:])
    print("=========================\n")

    for i in range(raw_LCMS.getNrSpectra()):
        if raw_LCMS[i].getMSLevel() == 1:
            rt_total = raw_LCMS[i].getRT()

            _, _, x_comp = get_binned_xcomp(
                rt_total=rt_total,
                modulation_period=modulation_period,
                rt1_bin_width=rt1_bin_width,
                rt2_bin_width=rt2_bin_width
            )

            if not (xcomp_start < x_comp < xcomp_end):
                continue

            mz, intensity = raw_LCMS[i].get_peaks()

            mz_binned = np.digitize(mz, mz_bins) - 1
            nan_idx = np.argwhere(mz_binned == -1)
            mz_binned = np.delete(mz_binned, nan_idx)
            intensity = np.delete(intensity, nan_idx)

            grouped_intensity = [np.max([c for _, c in g]) for _, g in groupby(zip(mz_binned, intensity), key=lambda x: x[0])]
            grouped_bins = np.unique(mz_binned)

            xcomp_binned = pd.cut([x_comp], xcomp_bins)
            xcomp_binned = xcomp_binned.rename_categories(list(range(0, len(xcomp_bins))))

            if np.sum(binned_matrix[xcomp_binned[0], :]) != 0:
                curr_val = binned_matrix[xcomp_binned[0], :]
                new_val = np.zeros((dim[1]))
                new_val[grouped_bins] = grouped_intensity
                binned_matrix[xcomp_binned[0], :] = np.max(np.stack((curr_val, new_val)), axis=0)
            else:
                binned_matrix[xcomp_binned[0], grouped_bins] = grouped_intensity

    if save_path is not None and os.path.exists(save_path):
        if filename is not None:
            np.save(os.path.join(save_path, str(filename) + '.npy'), binned_matrix)
        else:
            if filetype == "mzML":
                np.save(os.path.join(save_path, raw_LCMS.getLoadedFilePath().split('/')[-1][:-5] + '.npy'), binned_matrix)
            elif filetype == "mzdata":
                np.save(os.path.join(save_path, raw_LCMS.getLoadedFilePath().split('/')[-1][:-11] + '.npy'), binned_matrix)

        xcomp_axis_path = os.path.join(save_path, "xcomp_axis.npy")
        mz_axis_path = os.path.join(save_path, "mz_axis.npy")

        if not os.path.exists(xcomp_axis_path):
            np.save(xcomp_axis_path, xcomp_axis)

        if not os.path.exists(mz_axis_path):
            np.save(mz_axis_path, mz_axis)

    return binned_matrix


def __default_binning(
    raw_LCMS,
    dim=(856, 1024),
    xcomp_start=500.0,
    xcomp_end=1468.0,
    mz_start=40,
    mz_end=800,
    modulation_period=4.0,
    rt1_bin_width=2.0,
    rt2_bin_width=0.01,
    save_path=None,
    filename=None,
    filetype="mzML"
):
    """
    Default binning for GCxGC mzML.

    rows = x_comp bins
    cols = m/z bins
    x_comp = rt1_binned + (rt2_binned / modulation_period)
    """

    mz_bins = pd.interval_range(mz_start, mz_end, periods=dim[1])
    start = [pd.Interval(left=0, right=mz_bins[0].right)]
    end = [pd.Interval(left=mz_bins[-1].left, right=np.Inf)]
    mz_bins = pd.IntervalIndex(start + list(mz_bins[1:-1]) + end)

    xcomp_bins = pd.interval_range(xcomp_start, xcomp_end, periods=dim[0])
    xcomp_axis = build_xcomp_axis(xcomp_start, xcomp_end, dim[0])

    mz_axis = []
    for b in mz_bins:
        mz_axis.append((b.left + b.right) / 2.0)
    mz_axis = np.array(mz_axis, dtype=float)

    binned_matrix = np.zeros((dim[0], dim[1]))

    print("\n=== X_COMP DIAGNOSTIC ===")
    all_xcomp = []

    for i in range(raw_LCMS.getNrSpectra()):
        if raw_LCMS[i].getMSLevel() == 1:
            rt_total = raw_LCMS[i].getRT()

            _, _, x_comp = get_binned_xcomp(
                rt_total=rt_total,
                modulation_period=modulation_period,
                rt1_bin_width=rt1_bin_width,
                rt2_bin_width=rt2_bin_width
            )

            if xcomp_start < x_comp < xcomp_end:
                all_xcomp.append(x_comp)

    unique_xcomp = sorted(set(all_xcomp))
    print("Total MS1 scans in range:", len(all_xcomp))
    print("Unique x_comp positions:", len(unique_xcomp))
    print("First 10 x_comp:", unique_xcomp[:10])
    print("Last 10 x_comp:", unique_xcomp[-10:])
    print("=========================\n")

    for i in range(raw_LCMS.getNrSpectra()):
        if raw_LCMS[i].getMSLevel() == 1:
            rt_total = raw_LCMS[i].getRT()

            _, _, x_comp = get_binned_xcomp(
                rt_total=rt_total,
                modulation_period=modulation_period,
                rt1_bin_width=rt1_bin_width,
                rt2_bin_width=rt2_bin_width
            )

            if not (xcomp_start < x_comp < xcomp_end):
                continue

            mz, intensity = raw_LCMS[i].get_peaks()

            mz_binned = pd.cut(mz, mz_bins)
            mz_binned = mz_binned.rename_categories(list(range(0, dim[1])))
            nan_idx = np.argwhere(mz_binned.codes == -1)
            mz_binned = np.delete(mz_binned, nan_idx)
            intensity = np.delete(intensity, nan_idx)

            grouped_intensity = [np.max([c for _, c in g]) for _, g in groupby(zip(mz_binned, intensity), key=lambda x: x[0])]
            grouped_bins = np.unique(mz_binned)

            xcomp_binned = pd.cut([x_comp], xcomp_bins)
            xcomp_binned = xcomp_binned.rename_categories(list(range(0, dim[0])))

            if np.sum(binned_matrix[xcomp_binned[0], :]) != 0:
                curr_val = binned_matrix[xcomp_binned[0], :]
                new_val = np.zeros((dim[1]))
                new_val[grouped_bins] = grouped_intensity
                binned_matrix[xcomp_binned[0], :] = np.max(np.stack((curr_val, new_val)), axis=0)
            else:
                binned_matrix[xcomp_binned[0], grouped_bins] = grouped_intensity

    if save_path is not None and os.path.exists(save_path):
        if filename is not None:
            np.save(os.path.join(save_path, str(filename) + '.npy'), binned_matrix)
        else:
            if filetype == "mzML":
                np.save(os.path.join(save_path, raw_LCMS.getLoadedFilePath().split('/')[-1][:-5] + '.npy'), binned_matrix)
            elif filetype == "mzdata":
                np.save(os.path.join(save_path, raw_LCMS.getLoadedFilePath().split('/')[-1][:-11] + '.npy'), binned_matrix)

        xcomp_axis_path = os.path.join(save_path, "xcomp_axis.npy")
        mz_axis_path = os.path.join(save_path, "mz_axis.npy")

        if not os.path.exists(xcomp_axis_path):
            np.save(xcomp_axis_path, xcomp_axis)

        if not os.path.exists(mz_axis_path):
            np.save(mz_axis_path, mz_axis)

    return binned_matrix


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("RUNing in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("DONE in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))