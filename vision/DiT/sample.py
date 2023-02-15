import os
import random
import string

import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image


def get_all_files(path):
    # 列出所有文件和文件夹
    files = os.listdir(path)

    # 过滤掉文件夹
    files = [path + f for f in files if os.path.isfile(os.path.join(path, f))]

    return files


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


device = "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
class_count = 1000
for i in range(class_count):
    data_dir = "sample/npdata/%d/" % i
    all_np_data_file = get_all_files(data_dir)
    for np_data_file in all_np_data_file:
        data = np.load(np_data_file)
        samples = torch.from_numpy(data).to(device)
        samples = vae.decode(samples / 0.18215).sample
        save_dir = "sample/image/%d" % i
        mkdir_if_not_exist(save_dir)
        for idx, sample in enumerate(samples):
            save_image(sample, save_dir + "/sample%s.png" % (''.join(random.sample(string.ascii_letters, 8))), nrow=1,
                       normalize=True,
                       value_range=(-1, 1))
