import numpy as np
import scipy.io as scio

def generate_binary_array(m, n, d, p):
    binary_array = np.random.choice([0, 1], size=(m, n, d), p=[1-p, p])
    return binary_array

m = 256  # Number of rows
n = 256  # Number of columns
d = 8  # Depth of the array
p = 0.9  # Probability of a value being 1

mask = generate_binary_array(m, n, d, p)

print(mask.sum()/524288)

path = './test_datasets/mask/binary_iid_mask_0.9.mat'
scio.savemat(path, {'mask': mask})