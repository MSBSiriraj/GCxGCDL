import os
import pandas as pd
import numpy as np

# =========================
# SETTINGS
# =========================

input_folder = r"F:\Working ASCII\NorPD"
output_folder = r"D:\OneDrive\Projects\FF66\LCMS-Net\Peaktables\Min_detect_10\NorPD\NorPD_bin05"
meta_excel = r"D:\OneDrive\Projects\FF66\LCMS-Net\856_1024\Default_raw\NorPD\meta_NorPD.xlsx"

meta_sample_col = "Sample"
meta_index_col = "Index"
meta_group_col = "Group"
meta_group2_col = "Group2"   # optional

mz_bin_width = 0.05
rt1_bin_width = 2.0
rt2_bin_width = 0.02
modulation_period = 4.0
encoding = "latin1"
aggfunc = "max"   # "max" or "sum"

min_detection_fraction = 0.1
pseudocount = 1

output_csv = os.path.join(output_folder, "PT_NorPD_bin05_detect10.csv")
feature_list_csv = os.path.join(output_folder, "feature_list_PT_NorPD_bin05_detect10.csv")


# =========================
# FUNCTIONS
# =========================

def normalize_name(x):
    """
    Match CSV filename to meta Sample column.

    Examples:
    BKI-003_Result.csv -> BKI-003
    BP517_Result.csv   -> BP517
    """

    x = str(x).strip()

    x = x.replace(".csv", "").replace(".CSV", "")
    x = x.replace(".npy", "").replace(".NPY", "")

    # drop everything after first "_"
    if "_" in x:
        x = x.split("_")[0]

    return x


def load_meta(meta_path):
    meta = pd.read_excel(meta_path)
    meta.columns = meta.columns.str.strip()

    required_cols = [
        meta_sample_col,
        meta_index_col,
        meta_group_col
    ]

    missing_cols = [
        c for c in required_cols
        if c not in meta.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing columns in meta: {missing_cols}"
        )

    for col in required_cols:
        meta[col] = meta[col].astype(str).str.strip()

    has_group2 = meta_group2_col in meta.columns

    if has_group2:
        meta[meta_group2_col] = (
            meta[meta_group2_col]
            .astype(str)
            .str.strip()
        )

    meta["match_key"] = meta[meta_sample_col].apply(normalize_name)

    duplicated = meta[meta["match_key"].duplicated(keep=False)]

    if not duplicated.empty:
        print("\nERROR: duplicated Sample values in meta after matching.")
        cols_to_print = required_cols + ["match_key"]
        if has_group2:
            cols_to_print.insert(3, meta_group2_col)
        print(duplicated[cols_to_print])
        raise ValueError("Duplicated match keys in meta.")

    lookup_cols = [
        meta_index_col,
        meta_sample_col,
        meta_group_col
    ]

    if has_group2:
        lookup_cols.append(meta_group2_col)

    meta_lookup = (
        meta.set_index("match_key")[lookup_cols]
        .to_dict(orient="index")
    )

    return meta_lookup


def parse_peak_table(csv_file):
    df = pd.read_csv(csv_file, encoding=encoding)

    records = []

    for _, row in df.iterrows():
        rt1 = row.get("1st Dimension Time (s)")
        rt2 = row.get("2nd Dimension Time (s)")
        spectrum = row.get("Spectrum")

        if pd.isna(rt1) or pd.isna(rt2) or pd.isna(spectrum):
            continue

        try:
            rt1 = float(rt1)
            rt2 = float(rt2)
        except Exception:
            continue

        rt1_binned = round(
            round(rt1 / rt1_bin_width) * rt1_bin_width,
            6
        )

        rt2_binned = round(
            round(rt2 / rt2_bin_width) * rt2_bin_width,
            6
        )

        x_comp = rt1_binned + (rt2_binned / modulation_period)
        x_comp = round(float(x_comp), 6)

        for pair in str(spectrum).split():
            if ":" not in pair:
                continue

            mz_raw, intensity = pair.split(":")

            try:
                mz_raw = float(mz_raw)
                intensity = float(intensity)
            except Exception:
                continue

            mz_bin = round(
                round(mz_raw / mz_bin_width) * mz_bin_width,
                6
            )

            records.append({
                "x_comp": x_comp,
                "mz_bin": mz_bin,
                "intensity": intensity
            })

    return pd.DataFrame(records)


def convert_one_file(csv_file, sample_name):
    rec_df = parse_peak_table(csv_file)

    if rec_df.empty:
        return None

    grouped = rec_df.groupby(
        ["x_comp", "mz_bin"],
        as_index=False
    )["intensity"].agg(aggfunc)

    grouped["feature"] = (
        "x" + grouped["x_comp"].map(lambda v: f"{v:.6f}") +
        "_mz" + grouped["mz_bin"].map(lambda v: f"{v:.6f}")
    )

    row = grouped.set_index("feature")["intensity"]
    row.name = sample_name

    return row


# =========================
# MAIN
# =========================

def main():
    os.makedirs(output_folder, exist_ok=True)

    meta_lookup = load_meta(meta_excel)

    csv_files = [
        f for f in sorted(os.listdir(input_folder))
        if f.lower().endswith(".csv")
    ]

    if len(csv_files) == 0:
        print("No CSV files found.")
        return

    # =========================
    # SAFETY CHECK
    # =========================

    unmatched = []

    for filename in csv_files:
        match_key = normalize_name(filename)

        if match_key not in meta_lookup:
            unmatched.append((filename, match_key))

    if unmatched:
        print("\n==========================")
        print("ERROR: META MISMATCH")
        print("==========================")
        print("The following files could NOT be matched with meta Sample:")

        for filename, key in unmatched:
            print(f"{filename} -> {key}")

        print("\nNo output generated.")
        return

    print("\nSafety check passed.")
    print("CSV files found:", len(csv_files))

    # =========================
    # BUILD FEATURE ROWS
    # =========================

    sample_rows = []
    metadata_rows = []

    for filename in csv_files:
        csv_file = os.path.join(input_folder, filename)

        match_key = normalize_name(filename)
        meta_info = meta_lookup[match_key]

        sample_name = meta_info[meta_sample_col]

        msg = (
            f"Processing {filename}"
            f" | Index={meta_info[meta_index_col]}"
            f" | Sample={sample_name}"
            f" | Group={meta_info[meta_group_col]}"
        )

        if meta_group2_col in meta_info:
            msg += f" | Group2={meta_info[meta_group2_col]}"

        print(msg)

        try:
            row = convert_one_file(csv_file, sample_name)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

        if row is None:
            print(f"No valid peaks in {filename}")
            continue

        sample_rows.append(row)

        meta_row = {
            "Index": meta_info[meta_index_col],
            "Sample": meta_info[meta_sample_col],
            "Group": meta_info[meta_group_col],
        }

        if meta_group2_col in meta_info:
            meta_row["Group2"] = meta_info[meta_group2_col]

        metadata_rows.append(meta_row)

    if len(sample_rows) == 0:
        print("No valid peak tables were processed.")
        return

    # =========================
    # FEATURE TABLE
    # =========================

    feature_table = pd.DataFrame(sample_rows)
    feature_table = feature_table.fillna(0)

    # =========================
    # 70% DETECTION FILTER
    # =========================

    n_samples = feature_table.shape[0]

    min_detected_samples = max(
        2,
        int(np.ceil(min_detection_fraction * n_samples))
    )

    presence_count = (feature_table > 0).sum(axis=0)

    keep_features = presence_count[
        presence_count >= min_detected_samples
    ].index.tolist()

    print("\n=== FEATURE FILTER ===")
    print("Samples:", n_samples)
    print("Minimum detected samples:", min_detected_samples)
    print("Features before:", feature_table.shape[1])
    print("Features after:", len(keep_features))
    print("======================\n")

    feature_table = feature_table.loc[:, keep_features]

    # =========================
    # ADD PSEUDOCOUNT
    # =========================

    feature_table = feature_table + pseudocount

    # =========================
    # SORT + RENAME FEATURES
    # =========================

    feature_table = feature_table.reindex(
        sorted(feature_table.columns),
        axis=1
    )

    feature_mapping = pd.DataFrame({
        "NewFeature": [
            f"Met{i + 1}"
            for i in range(feature_table.shape[1])
        ],
        "OriginalFeature": feature_table.columns
    })

    feature_mapping.to_csv(feature_list_csv, index=False)

    feature_table.columns = feature_mapping["NewFeature"].tolist()

    print("Saved feature mapping:", feature_list_csv)

    # =========================
    # ADD METADATA COLUMNS
    # =========================

    meta_df = pd.DataFrame(metadata_rows)

    final_table = pd.concat(
        [
            meta_df.reset_index(drop=True),
            feature_table.reset_index(drop=True)
        ],
        axis=1
    )

    final_table.to_csv(output_csv, index=False)

    print("\nDONE")
    print("Saved:", output_csv)
    print("Shape:", final_table.shape)


if __name__ == "__main__":
    main()