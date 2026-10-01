import pandas as pd
import numpy as np
import pyopenms as oms
import os
import argparse
import time

from itertools import groupby


# ============================================================
# GCxGC binning script using elution time / total RT as xcomp_RT
# ============================================================
# Main change from the previous version:
#   Previous: rt_total -> rt1, rt2 -> x_comp = rt1_binned + rt2_binned / modulation_period
#   This one: xcomp_RT = rt_total directly
#
# Therefore, the row axis is simply the original elution time from mzML.
# xcomp_start and xcomp_end are still absolute RT values in seconds.
# ============================================================


def args_setting():
    parser = argparse.ArgumentParser(description='ArgUtils')
    parser.add_argument('-meta', type=str, dest='meta_filepath',
                        default=r"D:/OneDrive/Projects/FF66/LCMS-Net/extra_mzml/meta.xlsx")
    parser.add_argument('-data', type=str, dest='raw_folderpaths',
                        default=r"D:/OneDrive/Projects/FF66/LCMS-Net/extra_mzml")
    parser.add_argument('-save', dest='save_path',
                        default=r"D:/OneDrive/Projects/FF66/LCMS-Net/extra_mzml")
    parser.add_argument('-mode', type=str, dest='mode', default='default')

    # Kept for compatibility with older command lines, but NOT used in elution-time mode.
    parser.add_argument('-mod', type=float, dest='modulation_period', default=4.0)
    parser.add_argument('-rt1_bin', type=float, dest='rt1_bin_width', default=2.0)
    parser.add_argument('-rt2_bin', type=float, dest='rt2_bin_width', default=0.01)

    # These now define the total elution-time range, in seconds.
    parser.add_argument('-xcomp_start', type=float, dest='xcomp_start', default=500.0)
    parser.add_argument('-xcomp_end', type=float, dest='xcomp_end', default=1468.0)

    return parser.parse_args()


def get_elution_xcomp(rt_total):
    """Use raw elution time / total RT directly as xcomp_RT."""
    return float(rt_total)


def build_xcomp_axis(xcomp_start, xcomp_end, n_rows):
    """Axis centers for elution-time rows."""
    edges = np.linspace(xcomp_start, xcomp_end, n_rows + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return centers


def main():
    args = args_setting()

    meta_filepath = os.path.abspath(args.meta_filepath)
    folder = os.path.abspath(args.raw_folderpaths)
    save_path = os.path.abspath(args.save_path)

    mode = args.mode
    xcomp_start = args.xcomp_start
    xcomp_end = args.xcomp_end

    print("\n=== PATH CHECK ===")
    print("CURRENT WORKING DIR:", os.getcwd())
    print("META PATH:", meta_filepath)
    print("INPUT FOLDER:", folder)
    print("SAVE FOLDER:", save_path)
    print("META EXISTS:", os.path.exists(meta_filepath))
    print("INPUT EXISTS:", os.path.exists(folder))
    print("SAVE EXISTS:", os.path.exists(save_path))
    print("==================\n")

    data = pd.read_excel(meta_filepath, index_col=0, engine="openpyxl")
    data.index = data.index.map(str)

    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)

    filepaths = os.listdir(folder)
    processed_files = []

    if save_path is not None:
        processed_files = os.listdir(save_path)
        processed_files = [
            file.split(".")[0]
            for file in processed_files
            if file.endswith(".npy")
        ]

        # Avoid treating axis files as processed sample names if they exist.
        processed_files = [
            f for f in processed_files
            if f not in ["xcomp_axis", "mz_axis"]
        ]

        filepaths = [
            file for file in filepaths
            if (file.split(".")[0] not in processed_files)
            and (file.split(".")[0] in data.index)
        ]

    print("\n=== FILE FILTER CHECK ===")
    print("Total files in input folder:", len(os.listdir(folder)))
    print("Already processed npy:", len(processed_files))
    print("Files remaining to process:", len(filepaths))
    print("First 10 remaining files:", filepaths[:10])
    print("===========================\n")

    for file in filepaths:
        if file.endswith(".mzML"):
            sample = file.split(".")[0]

            if sample in data.index.tolist():
                filepath = os.path.abspath(os.path.join(folder, file))

                print("\n----------------------------------")
                print("Sample:", sample)
                print("Loading from:", filepath)
                print("File exists:", os.path.exists(filepath))
                print("----------------------------------")

                exp = oms.MSExperiment()

                try:
                    oms.MzMLFile().load(filepath, exp)

                    if mode == "adaptive":
                        __adaptive_binning(
                            exp,
                            save_path=save_path,
                            filename=sample,
                            filetype="mzML",
                            xcomp_start=xcomp_start,
                            xcomp_end=xcomp_end
                        )
                    else:
                        __default_binning(
                            exp,
                            save_path=save_path,
                            filename=sample,
                            filetype="mzML",
                            xcomp_start=xcomp_start,
                            xcomp_end=xcomp_end
                        )

                except Exception as e:
                    print(f"ERROR: File {file} could not be loaded! {e}")

        elif file.endswith(".mzdata.xml"):
            sample = file.split(".")[0]
            filepath = os.path.abspath(os.path.join(folder, file))

            print("\n----------------------------------")
            print("Sample:", sample)
            print("Loading from:", filepath)
            print("File exists:", os.path.exists(filepath))
            print("----------------------------------")

            exp = oms.MSExperiment()

            try:
                oms.MzDataFile().load(filepath, exp)

                if mode == "adaptive":
                    __adaptive_binning(
                        exp,
                        save_path=save_path,
                        filename=sample,
                        filetype="mzdata",
                        xcomp_start=xcomp_start,
                        xcomp_end=xcomp_end
                    )
                else:
                    __default_binning(
                        exp,
                        save_path=save_path,
                        filename=sample,
                        filetype="mzdata",
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
    save_path=None,
    filename=None,
    filetype="mzML"
):
    xcomp_bins = pd.interval_range(xcomp_start, xcomp_end, periods=dim[0])
    xcomp_axis = build_xcomp_axis(xcomp_start, xcomp_end, dim[0])

    mz_bins = []
    mz_bins_densities = [
        0.012, 0.048, 0.1, 0.128, 0.136, 0.099,
        0.05, 0.049, 0.048, 0.065, 0.026, 0.009,
        0.006, 0.045, 0.111, 0.063, 0.001, 0.001, 0.004
    ]

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

    print("\n=== ELUTION-TIME XCOMP DIAGNOSTIC ===")
    all_xcomp = []

    for i in range(raw_LCMS.getNrSpectra()):
        if raw_LCMS[i].getMSLevel() == 1:
            rt_total = raw_LCMS[i].getRT()
            x_comp = get_elution_xcomp(rt_total)

            if xcomp_start < x_comp < xcomp_end:
                all_xcomp.append(x_comp)

    unique_xcomp = sorted(set(all_xcomp))
    print("Total MS1 scans in range:", len(all_xcomp))
    print("Unique elution-time positions:", len(unique_xcomp))
    print("First 10 xcomp_RT:", unique_xcomp[:10])
    print("Last 10 xcomp_RT:", unique_xcomp[-10:])
    print("=====================================\n")

    for i in range(raw_LCMS.getNrSpectra()):
        if raw_LCMS[i].getMSLevel() == 1:
            rt_total = raw_LCMS[i].getRT()
            x_comp = get_elution_xcomp(rt_total)

            if not (xcomp_start < x_comp < xcomp_end):
                continue

            mz, intensity = raw_LCMS[i].get_peaks()

            mz_binned = np.digitize(mz, mz_bins) - 1
            nan_idx = np.argwhere(mz_binned == -1)
            mz_binned = np.delete(mz_binned, nan_idx)
            intensity = np.delete(intensity, nan_idx)

            grouped_intensity = [
                np.max([c for _, c in g])
                for _, g in groupby(zip(mz_binned, intensity), key=lambda x: x[0])
            ]
            grouped_bins = np.unique(mz_binned)

            xcomp_binned = pd.cut([x_comp], xcomp_bins)
            xcomp_binned = xcomp_binned.rename_categories(list(range(0, len(xcomp_bins))))
            row_idx = int(xcomp_binned[0])

            if np.sum(binned_matrix[row_idx, :]) != 0:
                curr_val = binned_matrix[row_idx, :]
                new_val = np.zeros((dim[1]))
                new_val[grouped_bins] = grouped_intensity
                binned_matrix[row_idx, :] = np.max(
                    np.stack((curr_val, new_val)),
                    axis=0
                )
            else:
                binned_matrix[row_idx, grouped_bins] = grouped_intensity

    _save_outputs(binned_matrix, xcomp_axis, mz_axis, save_path, filename)
    return binned_matrix


def __default_binning(
    raw_LCMS,
    dim=(856, 1024),
    xcomp_start=500.0,
    xcomp_end=1468.0,
    mz_start=40,
    mz_end=800,
    save_path=None,
    filename=None,
    filetype="mzML"
):
    mz_bins = pd.interval_range(mz_start, mz_end, periods=dim[1])
    start = [pd.Interval(left=0, right=mz_bins[0].right)]
    end = [pd.Interval(left=mz_bins[-1].left, right=np.inf)]
    mz_bins = pd.IntervalIndex(start + list(mz_bins[1:-1]) + end)

    xcomp_bins = pd.interval_range(xcomp_start, xcomp_end, periods=dim[0])
    xcomp_axis = build_xcomp_axis(xcomp_start, xcomp_end, dim[0])

    mz_axis = []
    for b in mz_bins:
        mz_axis.append((b.left + b.right) / 2.0)
    mz_axis = np.array(mz_axis, dtype=float)

    binned_matrix = np.zeros((dim[0], dim[1]))

    print("\n=== ELUTION-TIME XCOMP DIAGNOSTIC ===")
    all_xcomp = []

    for i in range(raw_LCMS.getNrSpectra()):
        if raw_LCMS[i].getMSLevel() == 1:
            rt_total = raw_LCMS[i].getRT()
            x_comp = get_elution_xcomp(rt_total)

            if xcomp_start < x_comp < xcomp_end:
                all_xcomp.append(x_comp)

    unique_xcomp = sorted(set(all_xcomp))
    print("Total MS1 scans in range:", len(all_xcomp))
    print("Unique elution-time positions:", len(unique_xcomp))
    print("First 10 xcomp_RT:", unique_xcomp[:10])
    print("Last 10 xcomp_RT:", unique_xcomp[-10:])
    print("=====================================\n")

    for i in range(raw_LCMS.getNrSpectra()):
        if raw_LCMS[i].getMSLevel() == 1:
            rt_total = raw_LCMS[i].getRT()
            x_comp = get_elution_xcomp(rt_total)

            if not (xcomp_start < x_comp < xcomp_end):
                continue

            mz, intensity = raw_LCMS[i].get_peaks()

            mz_binned = pd.cut(mz, mz_bins)
            mz_binned = mz_binned.rename_categories(list(range(0, dim[1])))
            nan_idx = np.argwhere(mz_binned.codes == -1)
            mz_binned = np.delete(mz_binned, nan_idx)
            intensity = np.delete(intensity, nan_idx)

            grouped_intensity = [
                np.max([c for _, c in g])
                for _, g in groupby(zip(mz_binned, intensity), key=lambda x: x[0])
            ]
            grouped_bins = np.unique(mz_binned)

            xcomp_binned = pd.cut([x_comp], xcomp_bins)
            xcomp_binned = xcomp_binned.rename_categories(list(range(0, dim[0])))
            row_idx = int(xcomp_binned[0])

            if np.sum(binned_matrix[row_idx, :]) != 0:
                curr_val = binned_matrix[row_idx, :]
                new_val = np.zeros((dim[1]))
                new_val[grouped_bins] = grouped_intensity
                binned_matrix[row_idx, :] = np.max(
                    np.stack((curr_val, new_val)),
                    axis=0
                )
            else:
                binned_matrix[row_idx, grouped_bins] = grouped_intensity

    _save_outputs(binned_matrix, xcomp_axis, mz_axis, save_path, filename)
    return binned_matrix


def _save_outputs(binned_matrix, xcomp_axis, mz_axis, save_path, filename):
    if save_path is not None and os.path.exists(save_path):
        if filename is None:
            raise ValueError("filename must be provided to avoid OpenMS path confusion.")

        save_file = os.path.join(save_path, str(filename) + ".npy")
        print("Saving to:", save_file)
        np.save(save_file, binned_matrix)

        xcomp_axis_path = os.path.join(save_path, "xcomp_axis.npy")
        mz_axis_path = os.path.join(save_path, "mz_axis.npy")

        if not os.path.exists(xcomp_axis_path):
            print("Saving xcomp_axis to:", xcomp_axis_path)
            np.save(xcomp_axis_path, xcomp_axis)

        if not os.path.exists(mz_axis_path):
            print("Saving mz_axis to:", mz_axis_path)
            np.save(mz_axis_path, mz_axis)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    print("RUNNING in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("DONE in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
