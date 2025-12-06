import os
import re
import hydra
from omegaconf import OmegaConf
import numpy as np
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
                input_data_dict["input"].append(
                    os.path.join(root, file)
                    )
    df = pd.DataFrame(input_data_dict)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y.%m.%d_%H.%M.%S")

    return df


def merge_dataframe(left, right, key="timestamp", how="inner"):
    right["timestamp"] = pd.to_datetime(right["timestamp"], format="%Y-%m-%d %H:%M:%S")
    df = pd.merge(left, right, on=key, how=how)
    df.sort_values(by="timestamp", inplace=True)
    return df


@hydra.main(
    config_path="../../configs/", 
    config_name="alexnet_helioviewer_config.yaml"
)
def main(cfg):

    # load huffingface dataset
    ds = load_dataset(cfg.data.repo)
    df_train = ds["train"].to_pandas()
    df_val_leaky = ds["leaky_validation"].to_pandas()
    df_val = ds["validation"].to_pandas()
    df_test = ds["test"].to_pandas()

    # load dataframe of input data 
    df_input = create_input_data_df(
        data_dir=cfg.data.input.path,
        file_ext=cfg.data.input.ext
        )
    
    # merge two dataframe
    df_train = merge_dataframe(df_input, df_train)
    df_val = merge_dataframe(df_input, df_val)
    df_val_leaky = merge_dataframe(df_input, df_val_leaky)
    df_test = merge_dataframe(df_input, df_test)
    df_val = pd.concat([df_val, df_val_leaky], axis=0)

    df_train.to_csv(cfg.data.index_file_path + "/train.csv", index=False)
    df_val.to_csv(cfg.data.index_file_path + "/validation.csv", index=False)
    df_test.to_csv(cfg.data.index_file_path + "/test.csv", index=False)


if __name__ == "__main__":
    main()
