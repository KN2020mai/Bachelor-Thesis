import os
import os.path as osp
import numpy as np
import pandas as pd
import cv2
import torch
import fastremap

def fix_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
fix_all_seeds(2021)

base_dir_sartorius = "../Data/Kaggle"
image_dir_sartorius = osp.join(base_dir_sartorius, "train")
train_csv_sartorius = pd.read_csv(osp.join(base_dir_sartorius, "train.csv"))
train_csv_gb_sartorius = train_csv_sartorius.groupby('id')
image_ids = np.array(train_csv_sartorius["id"].unique())
iperm = np.random.permutation(len(image_ids))
num_train_samples = int(len(image_ids) * 0.7)
num_val_samples = int(len(image_ids) * 0.1)
num_test_samples = len(image_ids) - num_train_samples - num_val_samples
train_image_ids = image_ids[iperm[:num_train_samples]]
val_image_ids = image_ids[iperm[num_train_samples:num_train_samples + num_val_samples]]
test_image_ids = image_ids[iperm[num_train_samples + num_val_samples:]]

train_csv_sartorius[train_csv_sartorius.id.isin(train_image_ids)].to_csv(osp.join(base_dir_sartorius, "sartorius_train.csv"), index=False)
train_csv_sartorius[train_csv_sartorius.id.isin(val_image_ids)].to_csv(osp.join(base_dir_sartorius, "sartorius_val.csv"), index=False)
train_csv_sartorius[train_csv_sartorius.id.isin(test_image_ids)].to_csv(osp.join(base_dir_sartorius, "sartorius_test.csv"), index=False)
