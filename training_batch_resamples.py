import platform
import os
import time
import argparse
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import create_model
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from utils import load_data, utils


if platform.system().lower() == 'linux':
    print("[INFO] run in linux platform!")


def str2bool(v):
    """Safer bool parser for command-line arguments."""
    if isinstance(v, bool):
        return v
    if str(v).lower() in ("yes", "true", "t", "1", "y"):
        return True
    if str(v).lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected: True/False")


def args_setting():
    """Parse arguments from command line and set default values."""

    parser = argparse.ArgumentParser(description='ArgUtils')

    parser.add_argument('-meta', type=str, dest='meta_filepath',
                        default=r"D:\OneDrive\Projects\FF66\LCMS-Net\856_1024\Default_raw\xcomp_RT\All_HD_PD\meta_All_HD_PD.xlsx")

    parser.add_argument('-data', type=str, dest='raw_folderpaths',
                        default=r"F:\Working npy\1024\Default_Raw\ElutionRT\RT_random\All_HD_PD\RT_random5")

    parser.add_argument('-train_list', type=str, dest='train_list_txt',
                        default=r"D:\OneDrive\Projects\FF66\LCMS-Net\856_1024\Default_raw\xcomp_RT\All_HD_PD\TrainList_biclass.txt",
                        help='Path to TrainList_biclass.txt containing Resample01, Resample02, etc.')

    # Batch resample range. This will run Resample01 ... Resample10 by default.
    parser.add_argument('-resample_start', type=int, dest='resample_start', default=1)
    parser.add_argument('-resample_end', type=int, dest='resample_end', default=10)

    # This is the BASE output folder. The script will create result_resample01, result_resample02, etc. inside it.
    parser.add_argument('-save', type=str, dest='save_dir',
                        default=r"D:\OneDrive\Projects\FF66\LCMS-Net\856_1024\Default_raw\ElutionRT\RT_random\All_HD_PD\RT_random5\Rerun")

    parser.add_argument('-handleImbalance', type=str, dest='handle_imbalance', default="ros",
                        choices=["ros", "class_weights", "none"])
    parser.add_argument('-norm', type=str2bool, dest='use_norm', default=True)
    parser.add_argument('-aug', type=str2bool, dest='use_aug', default=True)
    parser.add_argument('-lr', type=float, dest='lr', default=0.0001)
    parser.add_argument('-batch', type=int, dest='batch_size', default=4)
    parser.add_argument('-epoch', type=int, dest='epoch', default=200)

    args = parser.parse_args()
    return args


def get_run_log(history):
    """Create a dataframe with the training history of a model."""

    run_log = pd.DataFrame()
    run_log['epoch'] = list(range(1, 1 + len(history.history['loss'])))
    run_log['loss'] = history.history['loss']
    run_log['accuracy'] = history.history['accuracy']
    run_log['f1_score'] = history.history['f1_score']
    run_log['AUC-ROC'] = history.history['AUC-ROC']
    run_log['AUC-PR'] = history.history['AUC-PR']
    run_log['val_loss'] = history.history['val_loss']
    run_log['val_accuracy'] = history.history['val_accuracy']
    run_log['val_f1_score'] = history.history['val_f1_score']
    run_log['val_AUC-ROC'] = history.history['val_AUC-ROC']
    run_log['val_AUC-PR'] = history.history['val_AUC-PR']

    return run_log


# ============================================================
# SAVE TRAINING SETTINGS AND SAMPLE LISTS
# ============================================================

def save_training_settings(save_dir, args, resample_name, train_indices=None,
                           samples_pool=None, samples_train=None, samples_val=None, samples_test=None):
    """Save tunable training settings, resample info, and sample lists for reproducibility."""

    os.makedirs(save_dir, exist_ok=True)
    settings_path = os.path.join(save_dir, "training_settings.txt")

    with open(settings_path, "w") as f:
        f.write("=== TRAINING SETTINGS ===\n\n")

        f.write("[Resample]\n")
        f.write(f"Resample name: {resample_name}\n")
        f.write(f"TrainList indices count: {len(train_indices) if train_indices is not None else 'NA'}\n")
        f.write(f"Training pool samples count: {len(samples_pool) if samples_pool is not None else 'NA'}\n")
        f.write(f"Final train samples count: {len(samples_train) if samples_train is not None else 'NA'}\n")
        f.write(f"Validation samples count: {len(samples_val) if samples_val is not None else 'NA'}\n")

        f.write("\n[Arguments]\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")

        f.write("\n[Notes]\n")
        f.write("This file records tunable settings used for this model run.\n")
        f.write("The TrainList resample was used as the training pool, then split internally into train/validation.\n")

    if train_indices is not None:
        with open(os.path.join(save_dir, "train_indices_from_TrainList.txt"), "w") as f:
            f.write(",".join(str(x) for x in train_indices))

    if samples_pool is not None:
        with open(os.path.join(save_dir, "sample_pool_from_TrainList.txt"), "w") as f:
            f.write("\n".join([str(x) for x in samples_pool]))

    if samples_train is not None:
        with open(os.path.join(save_dir, "training.txt"), "w") as f:
            f.write("\n".join([str(x) for x in samples_train]))

    if samples_val is not None:
        with open(os.path.join(save_dir, "validation.txt"), "w") as f:
            f.write("\n".join([str(x) for x in samples_val]))

    if samples_test is not None:
        with open(os.path.join(save_dir, "test.txt"), "w") as f:
            f.write("\n".join([str(x) for x in samples_test]))

    print(f"[INFO] Saved training settings to: {settings_path}")


def load_resample_indices(txt_path, resample_name):
    """Read one resample line from TrainList_biclass.txt."""

    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.lower().startswith(resample_name.lower()):
            idx_str = line.split(":", 1)[1]
            return [int(x.strip()) for x in idx_str.split(",") if x.strip()]

    raise ValueError(f"{resample_name} not found in {txt_path}")


def prepare_meta(meta_filepath):
    """Load and clean meta.xlsx once."""

    meta = pd.read_excel(meta_filepath, engine="openpyxl")

    meta["Index"] = meta["Index"].astype(str).str.strip().astype(int)
    meta["Group"] = meta["Group"].astype(str).str.strip()
    meta["filename"] = meta["filename"].astype(str).str.strip()

    # filename in meta may or may not contain .npy
    meta["sample_name"] = (
        meta["filename"]
        .astype(str)
        .str.strip()
        .str.replace(".npy", "", regex=False)
    )

    return meta


def run_one_resample(args, meta_original, resample_num):
    """Run one LCMS-Net training job for one resample."""

    resample_name = f"Resample{resample_num:02d}"
    save_dir = os.path.join(args.save_dir, f"result_resample{resample_num:02d}")
    os.makedirs(save_dir, exist_ok=True)

    # Create args dataframe after adding current resample info for reproducibility.
    args.current_resample = resample_name
    args.current_save_dir = save_dir
    args_df = utils.create_args_df(args)

    meta = meta_original.copy()
    raw_folderpaths = [args.raw_folderpaths]
    input_shape = (856, 1024)

    use_norm = args.use_norm
    use_aug = args.use_aug
    handle_imbalance = args.handle_imbalance
    lr = args.lr
    epochs = args.epoch
    batch_size = args.batch_size

    train_indices = load_resample_indices(args.train_list_txt, resample_name)

    # Map TrainList indices -> filenames/sample names.
    selected_samples = meta.loc[
        meta["Index"].isin(train_indices),
        "sample_name"
    ].tolist()

    if len(selected_samples) == 0:
        raise ValueError(f"No samples selected for {resample_name}. Check that TrainList indices match meta['Index'].")

    # Make filename/sample_name the dataframe index.
    meta = meta.set_index("sample_name")

    # Subset meta for this resample training pool.
    meta_selected = meta.loc[selected_samples].copy()
    num_classes = len(np.unique(meta_selected["Group"]))

    # Create temporary LCMS-Net-compatible meta for load_data.data_gen.
    meta_for_loader = meta_selected[["Group"]].copy()
    temp_meta_path = os.path.join(save_dir, "meta_selected_for_loader.xlsx")
    meta_for_loader.to_excel(temp_meta_path)
    meta_filepath_for_loader = temp_meta_path

    print("\n" + "=" * 70)
    print(f"=== STARTING {resample_name} ===")
    print("Output folder:", save_dir)
    print("Train indices count:", len(train_indices))
    print("Selected samples:", len(selected_samples))
    print("First 10 selected samples:", selected_samples[:10])
    print("Groups:")
    print(meta_selected["Group"].value_counts())
    print("=" * 70 + "\n")

    print("\n=== DEBUG META SELECTION ===")
    print("Resample:", resample_name)
    print("Train indices count:", len(train_indices))
    print("First 20 train indices:", train_indices[:20])
    print("Meta shape:", meta.shape)
    print("Meta columns:", meta.columns.tolist())
    print("First 20 meta Index:", meta["Index"].head(20).tolist())
    print("Selected samples count:", len(selected_samples))
    print("First 20 selected samples:", selected_samples[:20])
    print("Meta selected shape:", meta_selected.shape)
    print("============================\n")

    samples_train, samples_val = utils.split_samples(
        meta_selected.index.astype(str),
        meta_selected["Group"],
        split_size=0.30
    )
    samples_train = [str(x) for x in samples_train]
    samples_val = [str(x) for x in samples_val]
    random.shuffle(samples_train)

    class_weights = None
    if handle_imbalance == "ros":
        ros = RandomOverSampler()
        samples_train_resampled, y_resampled = ros.fit_resample(
            np.array(samples_train).reshape(-1, 1),
            meta_selected.loc[samples_train]["Group"].to_list()
        )
        samples_train = samples_train_resampled.flatten().astype(str).tolist()
        random.shuffle(samples_train)

    elif handle_imbalance == "class_weights":
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(meta_selected["Group"]),
            y=meta_selected["Group"].tolist()
        )
        class_weights = dict(enumerate(weights))

    elif handle_imbalance == "none":
        class_weights = None

    else:
        raise ValueError("handleImbalance must be 'ros', 'class_weights', or 'none'")

    with tf.device('/CPU:0'):
        val_dataset = tf.data.Dataset.from_generator(
            load_data.data_gen,
            args=(raw_folderpaths, meta_filepath_for_loader, samples_val, use_norm, False),
            output_signature=(
                tf.TensorSpec(shape=input_shape, dtype=tf.float64),
                tf.TensorSpec(shape=(num_classes,), dtype=tf.int32)
            )
        )
        val_dataset = val_dataset.batch(batch_size, drop_remainder=False).prefetch(1).cache()

        train_dataset = tf.data.Dataset.from_generator(
            load_data.data_gen,
            args=(raw_folderpaths, meta_filepath_for_loader, samples_train, use_norm, use_aug),
            output_signature=(
                tf.TensorSpec(shape=input_shape, dtype=tf.float64),
                tf.TensorSpec(shape=(num_classes,), dtype=tf.int32)
            )
        )
        train_dataset = train_dataset.batch(batch_size, drop_remainder=False).prefetch(3)

    model = create_model.create_model(input_shape=input_shape, classes=num_classes)
    model = create_model.compile_model(model, lr=lr)

    # ============================================================
    # CHECK LABELS INSIDE TENSORFLOW DATASETS
    # ============================================================

    y_train_check = np.concatenate([y for x, y in train_dataset], axis=0)
    y_val_check = np.concatenate([y for x, y in val_dataset], axis=0)

    print("\n=== DATASET LABEL CHECK ===")
    print("Resample:", resample_name)
    print("y_train shape:", y_train_check.shape)
    print("y_train first 10:", y_train_check[:10])
    print("y_train class counts:", y_train_check.sum(axis=0))
    print("y_val shape:", y_val_check.shape)
    print("y_val first 10:", y_val_check[:10])
    print("y_val class counts:", y_val_check.sum(axis=0))
    print("===========================\n")

    early_stopping = EarlyStopping(
        monitor='val_f1_score',
        patience=10,
        start_from_epoch=3,
        restore_best_weights=True,
        mode='max'
    )
    lr_schedule = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=0.0000001
    )

    history = model.fit(
        train_dataset,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[lr_schedule, early_stopping],
        verbose=1,
        class_weight=class_weights
    )
    run_log = get_run_log(history)

    print('[INFO] run log:')
    print(run_log)

    # Save model and logs.
    model_id = str(int(time.time())) + str(np.random.randint(0, 1000))
    model_savedir = os.path.join(save_dir, f"CNN_{model_id}")
    os.makedirs(model_savedir, exist_ok=True)

    args_path = os.path.join(model_savedir, 'args.txt')
    args_df.to_csv(args_path, index=False, sep='\t')

    log_path = os.path.join(model_savedir, 'log.txt')
    run_log.to_csv(log_path, index=False, sep='\t')

    plt.plot(run_log["epoch"], run_log["loss"], label='Train')
    plt.plot(run_log["epoch"], run_log["val_loss"], label='Valid')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(model_savedir, "loss.png"))
    plt.close()

    plt.plot(run_log["epoch"], run_log["f1_score"], label='Train')
    plt.plot(run_log["epoch"], run_log["val_f1_score"], label='Valid')
    plt.xlabel("Epoch")
    plt.ylabel("F1-Score")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(model_savedir, "f1_score.png"))
    plt.close()

    utils.eval(
        model,
        train_dataset,
        np.concatenate([y for x, y in train_dataset], axis=0),
        np.unique(meta_selected["Group"]),
        "CNN_train",
        model_savedir
    )
    utils.eval(
        model,
        val_dataset,
        np.concatenate([y for x, y in val_dataset], axis=0),
        np.unique(meta_selected["Group"]),
        "CNN_val",
        model_savedir
    )

    model_path = os.path.join(model_savedir, 'model.h5')
    model.save(model_path)

    utils.save_split_data(samples_train, samples_val, model_path.replace('.h5', ''))

    save_training_settings(
        save_dir=model_savedir,
        args=args,
        resample_name=resample_name,
        train_indices=train_indices,
        samples_pool=selected_samples,
        samples_train=list(samples_train),
        samples_val=list(samples_val),
        samples_test=None
    )

    # Clear memory before the next resample.
    tf.keras.backend.clear_session()

    print(f"\n[INFO] FINISHED {resample_name}")
    print(f"[INFO] Saved model folder: {model_savedir}\n")


def main():
    args = args_setting()

    if args.resample_end < args.resample_start:
        raise ValueError("resample_end must be >= resample_start")

    meta_original = prepare_meta(args.meta_filepath)

    print("\n=== BATCH RESAMPLE RUN ===")
    print("Meta:", args.meta_filepath)
    print("Data:", args.raw_folderpaths)
    print("TrainList:", args.train_list_txt)
    print("Save base folder:", args.save_dir)
    print(f"Running Resample{args.resample_start:02d} to Resample{args.resample_end:02d}")
    print("==========================\n")

    failed = []

    for resample_num in range(args.resample_start, args.resample_end + 1):
        try:
            run_one_resample(args, meta_original, resample_num)
        except Exception as e:
            resample_name = f"Resample{resample_num:02d}"
            failed.append((resample_name, str(e)))
            print("\n" + "!" * 70)
            print(f"[ERROR] {resample_name} failed:")
            print(e)
            print("Continuing to next resample.")
            print("!" * 70 + "\n")
            tf.keras.backend.clear_session()

    print("\n=== BATCH RUN SUMMARY ===")
    if len(failed) == 0:
        print("All resamples completed successfully.")
    else:
        print("Failed resamples:")
        for resample_name, error_msg in failed:
            print(f"- {resample_name}: {error_msg}")
    print("=========================\n")


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("[INFO] RUNNING in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    main()
    print("[INFO] END in %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
