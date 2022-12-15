# Bachelor Thesis: Instance Segmentation using Flow Tracking

## Abstract
Microscopy is a key technique adopted in biological and biomedical studies to capture cell appearance.
varieties of advanced microscopy techniques are applied for in situ, in vivo or in vitro imaging at different scales, such as brightfield and fluorescence microscopy.
However, systematic manual analysis is prohibitively time-consuming, which rises the demand for high-throughput and automated microscopy image analysis.
With the assistance of computer vision techniques, automated identifying the cell bodies, membranes and nuclei from microscopy images can be achieved via common tasks, such as detection,
Image segmentation is a critical and important step, as quantifying the cell morphology is the key to a wide range of studies and applications, for example, further identifying the cell biological phenotype.
Recent years, deep learning-based techniques, especially Convolutional Neural Networks (CNNs), have extensively applied to and progressed substantially in computer vision.
In the thesis, we focus on developing deep learning approaches for cell instance segmentation on neuronal microscopy images.
The objective of this thesis is to develop a deep learning based framework for biomedical image instance segmentation.
More specifically, the thesis focuses on the bottom-up methods with a post-processing procedure by flow tracking.
We implemented models with different network backbones based on a flow tracking based bottom-up method so-called Cellpose and proposed two approaches, i.e., spatial attention mechanism and contour-aware loss functions, to improve both semantic and instance segmentation performance.

## Data
We use Sartorius kaggle competition dataset to evaluate our proposed methods. The followings are the instructions to download the data from kaggle and split data into train/val/test three parts.

1. Go to the 'Account' tab of your user profile (https://www.kaggle.com/USER_NAME/account) and select 'Create API Token' (Please replace the parameter USER_NAME with your own kaggle user name

2. A file containing your API credentials will be downloaded (kaggle.json)

3. Create a folder to store configuration file named “.kaggle”
      ```bash
      mkdir .kaggle
      ```

4. Move the file to the created folder
      ```bash
      mv kaggle.json .kaggle/
      ```

5. Grant read access to your credential file
      ```bash
      chmod 600 .kaggle/kaggle.json
      ```

6. Download the dataset

   ```bash
   kaggle competitions download -c sartorius-cell-instance-segmentation
   ```
   
   If you run into a kaggle: command not found error, then use the following command
   
   ```bash
   ~/.local/bin/kaggle competitions download -c sartorius-cell-instance-segmentation
   ```
