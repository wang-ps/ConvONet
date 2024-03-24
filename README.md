# Evaluation Metrics of Dual Octree Graph Networks

This repository contains the evaluation metrics of our paper
[Dual Octree Graph Networks](https://arxiv.org/abs/2205.02825), which are implemented by
[Convolutional Occupancy Networks](https://github.com/autonomousvision/convolutional_occupancy_networks).


## Installation

- Create an anaconda environment called `pytorch-1.4.0` using
    ```
    conda env create -f environment.yaml
    conda activate pytorch-1.4.0
   ```

- Compile the extension modules.
    ```
    python setup.py build_ext --inplace
    ```

## Evaluation

Denote the folder where you clone the code of our dual octree graph networks as `ognn`.

- Evaluate the results on the testing dataset of ShapeNet.

  ```bash
  python eval_meshes.py  \
        configs/pointcloud/shapenet.yaml  \
        --dataset_folder /ognn/data/ShapeNet/dataset  \
        --generation_dir /ognn/logs/shapenet_eval/test
  ```

- Evaluate the results on the unseen 5 categories of ShapeNet.

  ```bash
  python eval_meshes.py  \
        configs/pointcloud/shapenet.yaml  \
        --dataset_folder /ognn/data/ShapeNet/dataset.unseen5  \
        --generation_dir /ognn/logs/shapenet_eval/unseen5
  ```

- Evaluate the results on the synthetic room dataset.

  ```bash
  python eval_meshes.py  \
        configs/pointcloud/room.yaml  \
        --dataset_folder /ognn/data/room/synthetic_room_dataset  \
        --generation_dir /ognn/logs/docnn/room_eval/room
  ```
