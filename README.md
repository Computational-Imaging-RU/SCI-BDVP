# Untrained Neural Nets for Snapshot Compressive Imaging: Theory and Algorithms

## 1. Datasets and Masks



The greyscale datasets is under `test_datasets/simulation/`

Generate mask with different distribution with 
```
python generate_iid.py
```


## 2. Training

### Run the SCI-BDVP algorithm for recovering video snapshot compressive imaging (SCI):

#### 2.1. Noise free measurements reconstructioon with Generalized Alternative Projection (GAP):

```
python test_iterative.py --meas_noise 0 --denoise_method "GAP_dip" --step_size 1.0 --mask_path 'test_datasets/mask/binary_iid_mask_0.5.mat'
```

#### 2.2. Noisy measurements reconstructioon with Projected Gradient Descent (PGD):

```
python test_iterative.py --meas_noise 0.1 --denoise_method "GD_dip" --step_size 0.1 --mask_path 'test_datasets/mask/binary_iid_mask_0.5.mat'
```

## 3. Citations

<!-- Mengyu Zhao, Xi Chen, Xin Yuan, and Shirin Jalali. "Untrained Neural Nets for Snapshot Compressive Imaging: Theory and Algorithms." arXiv preprint arXiv: (2024). [paper](https://) -->

```shell
@misc{
}
```

