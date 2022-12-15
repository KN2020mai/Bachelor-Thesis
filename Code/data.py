import os
import os.path as osp
import numpy as np
import pandas as pd
import cv2
import torch
import fastremap

from dynamics import masks_to_flows, masks_to_tmaps

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

train_csv_sartorius = pd.read_csv(osp.join(base_dir_sartorius, "sartorius_train.csv"))
train_csv_gb_sartorius = train_csv_sartorius.groupby('id')
val_csv_sartorius = pd.read_csv(osp.join(base_dir_sartorius, "sartorius_val.csv"))
val_csv_gb_sartorius = val_csv_sartorius.groupby('id')
test_csv_sartorius = pd.read_csv(osp.join(base_dir_sartorius, "sartorius_test.csv"))
test_csv_gb_sartorius = test_csv_sartorius.groupby('id')

train_image_ids = np.array(train_csv_sartorius["id"].unique())
val_image_ids = np.array(val_csv_sartorius["id"].unique())
test_image_ids = np.array(test_csv_sartorius["id"].unique())

train_cell_types = np.array([train_csv_gb_sartorius.get_group(id)["cell_type"].tolist()[0] for id in train_image_ids])
val_cell_types = np.array([val_csv_gb_sartorius.get_group(id)["cell_type"].tolist()[0] for id in val_image_ids])
test_cell_types = np.array([test_csv_gb_sartorius.get_group(id)["cell_type"].tolist()[0] for id in test_image_ids])

num_train_samples = len(train_image_ids)
num_val_samples = len(val_image_ids)
num_test_samples = len(test_image_ids)

def decode_rle(rle, input_shape):
    """
    Decode run-length encoded segmentation mask string into 2d array
    ----------
    @param: rle_mask (str): Run-length encoded segmentation mask string
    @param: shape (tuple): Height and width of the mask
    @return: mask [numpy.ndarray of shape (height, width)]: Decoded 2d segmentation mask
    """
    rle = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (rle[0:][::2], rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros((input_shape[0] * input_shape[1]), dtype=np.uint8)
    for start, end in zip(starts, ends):
        mask[start:end] = 1
    mask = mask.reshape(input_shape[0], input_shape[1])
    return mask

def build_cellpose_masks(annotations, input_shape):
    '''
    annotations: list of run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 to N - mask, 0 - background
    '''
    (height, width) = input_shape
    masks = np.zeros((height*width), dtype=np.uint8)
    for instance_idx in range(len(annotations)):
        s = annotations[instance_idx].split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            masks[lo:hi] = instance_idx + 1
    return masks.reshape(input_shape)

# def build_coco_masks(annotations, input_shape):
#     '''
#     @param: annotations: list of run-length as string formated (start length)
#     @param: input_shape: (height,width) of array to return
#     @return: numpy array, 1 to N - mask, 0 - background
#     '''
#     (height, width) = input_shape
#     masks = np.zeros((len(annotations), height, width), dtype=np.uint8)
#     for instance_idx in range(len(annotations)):
#         a_mask = decode_rle(annotations[instance_idx], input_shape)
#         masks[instance_idx, :, :] = a_mask
#     return masks

def get_train_data(use_tmap=True):
    data, labels, image_names, cell_types = [], [], [], []
    for idx in range(num_train_samples):
    # for idx in range(8):
        image_id = train_image_ids[idx]
        cell_type = train_cell_types[idx]
        df_data_idx = train_csv_gb_sartorius.get_group(image_id)
        annotations = df_data_idx['annotation'].tolist()
        image = cv2.imread(os.path.join(image_dir_sartorius, image_id + ".png"), cv2.IMREAD_GRAYSCALE)
        mask = build_cellpose_masks(annotations, (image.shape[0], image.shape[1]))
        data.append(image)
        image_names.append(image_id)
        labels.append(mask)
        cell_types.append(cell_type)
    nimg = len(labels)
    labels = [labels[n][np.newaxis, :, :] for n in range(nimg)]
    labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
    veci = [masks_to_flows(labels[n][0]) for n in range(nimg)]
    if use_tmap:
        tmaps = [masks_to_tmaps(labels[n][0])[np.newaxis, :, :] for n in range(nimg)]
        flows = [np.concatenate((labels[n], labels[n] > 0.5, veci[n], tmaps[n]), axis=0).astype(np.float32) for n in range(nimg)]
    else:
        flows = [np.concatenate((labels[n], labels[n] > 0.5, veci[n]), axis=0).astype(np.float32) for n in range(nimg)]
    return data, flows, image_names, cell_types


def get_val_data(use_tmap=True):
    data, labels, image_names, cell_types = [], [], [], []
    for idx in range(num_val_samples):
    # for idx in range(8):
        image_id = val_image_ids[idx]
        cell_type = val_cell_types[idx]
        df_data_idx = val_csv_gb_sartorius.get_group(image_id)
        annotations = df_data_idx['annotation'].tolist()
        image = cv2.imread(os.path.join(image_dir_sartorius, image_id + ".png"), cv2.IMREAD_GRAYSCALE)
        mask = build_cellpose_masks(annotations, (image.shape[0], image.shape[1]))
        data.append(image)
        image_names.append(image_id)
        labels.append(mask)
        cell_types.append(cell_type)
    nimg = len(labels)
    labels = [labels[n][np.newaxis, :, :] for n in range(nimg)]
    labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
    veci = [masks_to_flows(labels[n][0]) for n in range(nimg)]
    if use_tmap:
        tmaps = [masks_to_tmaps(labels[n][0])[np.newaxis, :, :] for n in range(nimg)]
        flows = [np.concatenate((labels[n], labels[n] > 0.5, veci[n], tmaps[n]), axis=0).astype(np.float32) for n in range(nimg)]
    else:
        flows = [np.concatenate((labels[n], labels[n] > 0.5, veci[n]), axis=0).astype(np.float32) for n in range(nimg)]
    return data, flows, image_names, cell_types

def get_test_data(use_tmap=True):
    data, labels, image_names, cell_types = [], [], [], []
    for idx in range(num_test_samples):
    # for idx in range(8):
        image_id = test_image_ids[idx]
        cell_type = test_cell_types[idx]
        df_data_idx = test_csv_gb_sartorius.get_group(image_id)
        annotations = df_data_idx['annotation'].tolist()
        image = cv2.imread(os.path.join(image_dir_sartorius, image_id + ".png"), cv2.IMREAD_GRAYSCALE)
        mask = build_cellpose_masks(annotations, (image.shape[0], image.shape[1]))
        data.append(image)
        image_names.append(image_id)
        labels.append(mask)
        cell_types.append(cell_type)
    nimg = len(labels)
    labels = [labels[n][np.newaxis, :, :] for n in range(nimg)]
    labels = [fastremap.renumber(label, in_place=True)[0] for label in labels]
    veci = [masks_to_flows(labels[n][0]) for n in range(nimg)]
    if use_tmap:
        tmaps = [masks_to_tmaps(labels[n][0])[np.newaxis, :, :] for n in range(nimg)]
        flows = [np.concatenate((labels[n], labels[n] > 0.5, veci[n], tmaps[n]), axis=0).astype(np.float32) for n in range(nimg)]
    else:
        flows = [np.concatenate((labels[n], labels[n] > 0.5, veci[n]), axis=0).astype(np.float32) for n in range(nimg)]
    return data, flows, image_names, cell_types