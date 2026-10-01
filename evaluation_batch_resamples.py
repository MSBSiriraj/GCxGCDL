import os
import re
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from utils import load_data, utils
from utils import load_model as loader


def args_setting():
    parser = argparse.ArgumentParser(description="LCMS-Net Batch Resample Eval_raw\Evaluation")

    parser.add_argument(
        "-meta",
        type=str,
        dest="meta_filepath",
        default=r"D:\OneDrive\Projects\FF66\LCMS-Net\856_1024\Default_raw\xcomp_RT\All_HD_PD\meta_All_HD_PD.xlsx"
    )

    parser.add_argument(
        "-data",
        type=str,
        dest="raw_folderpath",
        default=r"F:\Working npy\1024\Default_Raw\ElutionRT\RT_random\All_HD_PD\RT_random5"
    )

    parser.add_argument(
        "-testlist",
        type=str,
        dest="testlist_path",
        default=r"D:\OneDrive\Projects\FF66\LCMS-Net\856_1024\Default_raw\xcomp_RT\All_HD_PD\TestList_biclass.txt"
    )

    # IMPORTANT:
    # This should be the BASE folder that contains:
    #   result_resample01
    #   result_resample02
    #   ...

    #   result_resample10
    #
    # Do NOT point this to only result_resample10 if you want batch evaluation.
    parser.add_argument(
        "-models",
        type=str,
        dest="model_base_dir",
        default=r"D:\OneDrive\Projects\FF66\LCMS-Net\856_1024\Default_raw\ElutionRT\RT_random\All_HD_PD\RT_random5\Rerun"
    )

    parser.add_argument(
        "-save",
        type=str,
        dest="save_path",
        default=r"D:\OneDrive\Projects\FF66\LCMS-Net\856_1024\Default_raw\ElutionRT\RT_random\All_HD_PD\RT_random5\Rerun\Evaluation Batch"
    )

    parser.add_argument("-batch", type=int, dest="batch_size", default=4)
    parser.add_argument("-norm", type=bool, dest="use_norm", default=True)

    # Batch resample range
    parser.add_argument("-resample_start", type=int, dest="resample_start", default=1)
    parser.add_argument("-resample_end", type=int, dest="resample_end", default=10)

    # Folder naming pattern created by batch training script
    parser.add_argument(
        "-model_folder_prefix",
        type=str,
        dest="model_folder_prefix",
        default="result_resample"
    )

    return parser.parse_args()


def resample_label(i):
    """Return Resample01, Resample02, ..., Resample10."""
    return f"Resample{i:02d}"


def result_folder_name(i, prefix="result_resample"):
    """Return result_resample01, result_resample02, ..., result_resample10."""
    return f"{prefix}{i:02d}"


def load_resample_indices(txt_path, resample_name):

    with open(txt_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    for line in lines:

        clean = line.strip()

        if ":" not in clean:
            continue

        left, right = clean.split(":", 1)

        left = left.strip()

        if left.lower() == resample_name.lower():

            return [
                int(x.strip())
                for x in right.split(",")
                if x.strip()
            ]

    raise ValueError(
        f"{resample_name} not found in {txt_path}"
    )


def find_model_folders(model_dir):
    """
    Find CNN_* folders containing model.h5 inside one result_resampleXX folder.
    """

    if not os.path.isdir(model_dir):
        raise ValueError(f"Model folder does not exist: {model_dir}")

    model_folders = [
        os.path.join(model_dir, d)
        for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
        and os.path.exists(os.path.join(model_dir, d, "model.h5"))
    ]

    return sorted(model_folders)


def save_evaluation_settings(save_path, args):
    os.makedirs(save_path, exist_ok=True)

    settings_path = os.path.join(save_path, "evaluation_settings.txt")

    with open(settings_path, "w") as f:
        f.write("=== EVALUATION SETTINGS ===\n\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

    print(f"[INFO] Saved evaluation settings to: {settings_path}")


def main():
    args = args_setting()

    meta_filepath = args.meta_filepath
    raw_folderpaths = [args.raw_folderpath]
    testlist_path = args.testlist_path
    model_base_dir = args.model_base_dir
    save_path = args.save_path
    batch_size = args.batch_size
    use_norm = args.use_norm

    input_shape = (856, 1024)

    os.makedirs(save_path, exist_ok=True)
    save_evaluation_settings(save_path, args)

    # ============================================================
    # 1. READ AND CLEAN META
    # ============================================================
    meta = pd.read_excel(meta_filepath, engine="openpyxl")

    meta.columns = meta.columns.str.strip()
    meta["Index"] = meta["Index"].astype(str).str.strip().astype(int)
    meta["Group"] = meta["Group"].astype(str).str.strip()
    meta["filename"] = meta["filename"].astype(str).str.strip()

    meta["sample_name"] = (
        meta["filename"]
        .astype(str)
        .str.strip()
        .str.replace(".npy", "", regex=False)
    )

    meta = meta.set_index("sample_name")

    class_list = np.unique(meta["Group"])
    num_classes = len(class_list)

    # LCMS-Net loader needs meta indexed by sample name with Group column
    meta_for_loader = meta[["Group"]].copy()
    temp_meta_path = os.path.join(save_path, "meta_for_evaluation_loader.xlsx")
    meta_for_loader.to_excel(temp_meta_path)

    all_results = []

    # ============================================================
    # 2. LOOP THROUGH RESAMPLE01 TO RESAMPLE10
    # ============================================================
    for resample_i in range(args.resample_start, args.resample_end + 1):

        resample_name = resample_label(resample_i)
        result_dir_name = result_folder_name(resample_i, args.model_folder_prefix)
        model_dir = os.path.join(model_base_dir, result_dir_name)

        print("\n======================================")
        print("Evaluating:", resample_name)
        print("Model folder:", model_dir)
        print("======================================")

        try:
            # ----------------------------------------------------
            # Load matching TestList indices
            # ----------------------------------------------------
            test_indices = load_resample_indices(testlist_path, resample_name)

            test_samples = meta.loc[
                meta["Index"].isin(test_indices)
            ].index.astype(str).tolist()

            if len(test_samples) == 0:
                raise ValueError(f"No test samples selected for {resample_name}")

            print("Test indices count:", len(test_indices))
            print("Matched test samples:", len(test_samples))
            print("First 10 test samples:", test_samples[:10])
            print("Group counts:")
            print(meta.loc[test_samples, "Group"].value_counts())

            # ----------------------------------------------------
            # Find models inside this resample folder
            # ----------------------------------------------------
            model_folders = find_model_folders(model_dir)

            if len(model_folders) == 0:
                raise ValueError(f"No model.h5 files found inside {model_dir}")

            print("\nModels found:")
            for m in model_folders:
                print(" ", m)

            # ----------------------------------------------------
            # Create output folder for this resample
            # ----------------------------------------------------
            resample_save_path = os.path.join(save_path, resample_name)
            os.makedirs(resample_save_path, exist_ok=True)

            # Save test sample list
            pd.DataFrame({
                "sample_name": test_samples,
                "Group": meta.loc[test_samples, "Group"].values
            }).to_csv(
                os.path.join(resample_save_path, "samples_test.csv"),
                index=False
            )

            # ----------------------------------------------------
            # Build test dataset ONCE for this resample
            # ----------------------------------------------------
            with tf.device("/CPU:0"):
                test_dataset = tf.data.Dataset.from_generator(
                    load_data.data_gen,
                    args=(raw_folderpaths, temp_meta_path, test_samples, use_norm, False),
                    output_signature=(
                        tf.TensorSpec(shape=input_shape, dtype=tf.float64),
                        tf.TensorSpec(shape=(num_classes,), dtype=tf.int32)
                    )
                )

                test_dataset = (
                    test_dataset
                    .batch(batch_size, drop_remainder=False)
                    .prefetch(1)
                )

            y_true = np.concatenate([y for x, y in test_dataset], axis=0)

            # ----------------------------------------------------
            # Evaluate every model.h5 inside this result_resampleXX
            # ----------------------------------------------------
            for model_folder in model_folders:
                model_path = os.path.join(model_folder, "model.h5")
                model_name = os.path.basename(model_folder)

                model_save_path = os.path.join(resample_save_path, model_name)
                os.makedirs(model_save_path, exist_ok=True)

                print("\n--- Evaluating model ---")
                print("Resample:", resample_name)
                print("Model:", model_path)
                print("Save:", model_save_path)

                model = loader.SingleModel(model_path)

                utils.eval(
                    model,
                    test_dataset,
                    y_true,
                    class_list,
                    "CNN_test",
                    model_save_path,
                    False
                )

                all_results.append({
                    "resample": resample_name,
                    "model_folder": model_name,
                    "model_path": model_path,
                    "n_test_samples": len(test_samples),
                    "save_path": model_save_path,
                    "status": "OK",
                    "error": ""
                })

        except Exception as e:
            print(f"\n[ERROR] {resample_name} failed:")
            print(e)

            all_results.append({
                "resample": resample_name,
                "model_folder": "",
                "model_path": model_dir,
                "n_test_samples": "",
                "save_path": "",
                "status": "FAILED",
                "error": str(e)
            })

            # Continue to next resample instead of stopping the whole batch
            continue

    # ============================================================
    # 3. SAVE EVALUATION SUMMARY
    # ============================================================
    summary_df = pd.DataFrame(all_results)

    summary_path = os.path.join(save_path, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== BATCH EVALUATION COMPLETE ===")
    print(summary_df)
    print("Saved summary to:", summary_path)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    print("[INFO] RUNNING in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("[INFO] DONE in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
