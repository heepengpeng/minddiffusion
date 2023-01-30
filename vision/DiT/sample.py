import numpy as np
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image

device = "cpu"
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
data = np.load('data.npy')

samples = torch.from_numpy(data)
samples = vae.decode(samples / 0.18215).sample

# Save and display images:
save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
