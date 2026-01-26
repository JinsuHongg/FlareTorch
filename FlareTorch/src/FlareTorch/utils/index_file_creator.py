import os
import re
import hydra
import pandas as pd

from datasets import load_dataset


def extract_time_from_filename(file_name):
    match = re.search(r"\d{4}\.\d{2}\.\d{2}_\d{2}\.\d{2}\.\d{2}", file_name)
    if match:
        timestamp = match.group()
    return timestamp


def create_input_data_df(data_dir, file_ext):

    input_data_dict = {
        "timestamp": [],
        "input": [],
    }
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(file_ext):
                input_data_dict["timestamp"].append(extract_time_from_filename(file))
                input_data_dict["input"].append(os.path.join(root, file))
    df = pd.DataFrame(input_data_dict)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y.%m.%d_%H.%M.%S")

    return df


def map_cls_to_intensity(each_cls):
    mapping_dict = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}

    if each_cls == "FQ":
        return 1e-9
    else:
        goes_cls = each_cls[0]
        sub_cls = float(each_cls[1:])
        return mapping_dict[goes_cls] * sub_cls


def add_max_intensity(df):
    df["max_intensity"] = df["max_goes_class"].map(map_cls_to_intensity)
    return df


@hydra.main(config_path="../../../configs/", config_name="QR_resnet18_train.yaml")
def main(cfg):

    # load huffingface dataset
    ds = load_dataset(cfg.data.repo)
    df_train = ds["train"].to_pandas()
    df_val_leaky = ds["leaky_validation"].to_pandas()
    df_val = ds["validation"].to_pandas()
    df_test = ds["test"].to_pandas()

    # load dataframe of input data
    df_input = create_input_data_df(
        data_dir=cfg.data.input.path, file_ext=cfg.data.input.ext
    )
    # add intensity
    df_train = add_max_intensity(df_train)
    df_val_leaky = add_max_intensity(df_val_leaky)
    df_val = add_max_intensity(df_val)
    df_test = add_max_intensity(df_test)

    df_input.to_csv(cfg.data.index.path + "helioviewer_mag_input.csv", index=False)
    df_train.to_csv(cfg.data.index.path + "train.csv", index=False)
    df_val_leaky.to_csv(cfg.data.index.path + "leaky_validation.csv", index=False)
    df_val.to_csv(cfg.data.index.path + "validation.csv", index=False)
    df_test.to_csv(cfg.data.index.path + "test.csv", index=False)


if __name__ == "__main__":
    main()
