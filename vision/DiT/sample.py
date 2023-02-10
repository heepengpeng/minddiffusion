import random
import string

import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image

device = "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
class_count = 1000
for i in range(100):
    data_dir = "sample/0/data%d.npy" % i
    save_dir = "sample/image"
    data = np.load(data_dir)
    samples = torch.from_numpy(data)
    samples = vae.decode(samples / 0.18215).sample
    for idx, sample in enumerate(samples):
        save_image(sample, save_dir + "/sample%s.png" % (''.join(random.sample(string.ascii_letters, 8))), nrow=1,
                   normalize=True,
                   value_range=(-1, 1))