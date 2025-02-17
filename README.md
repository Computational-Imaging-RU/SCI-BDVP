# Untrained Neural Nets for Snapshot Compressive Imaging: Theory and Algorithms
[![Static Badge](https://img.shields.io/badge/NeurIPS_2024_paper-arxiv_link-blue)
](https://arxiv.org/abs/2406.03694) 
[![Static Badge](https://img.shields.io/badge/NeurIPS_2024_paper-Open_Review-red)
](https://openreview.net/forum?id=7aFEqIb1dp) 

## 1. Datasets and Masks



The greyscale datasets is under `test_datasets/simulation/`

Generate mask with different distribution with 
```
python generate_iid.py
```


## 2. Training

### Run the SCI-BDVP algorithm for recovering video snapshot compressive imaging (SCI):

#### 2.1. Noise free measurements reconstruction with generalized alternative projection (GAP):

```
python test_iterative.py --meas_noise 0 --denoise_method "GAP_dip" --step_size 1.0 --mask_path 'test_datasets/mask/binary_iid_mask_0.5.mat'
```

#### 2.2. Noisy measurements reconstruction with projected gradient descent (PGD):

```
python test_iterative.py --meas_noise 0.1 --denoise_method "GD_dip" --step_size 0.1 --mask_path 'test_datasets/mask/binary_iid_mask_0.5.mat'
```

## 3. Citations

<!-- Mengyu Zhao, Xi Chen, Xin Yuan, and Shirin Jalali. "Untrained Neural Nets for Snapshot Compressive Imaging: Theory and Algorithms." arXiv preprint arXiv: (2024). [paper](https://) -->

```shell
@misc{zhao2024untrained,
      title={Untrained Neural Nets for Snapshot Compressive Imaging: Theory and Algorithms}, 
      author={Mengyu Zhao and Xi Chen and Xin Yuan and Shirin Jalali},
      year={2024},
      eprint={2406.03694},
      archivePrefix={arXiv},
      primaryClass={id='cs.CV' full_name='Computer Vision and Pattern Recognition' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'}
}
```

