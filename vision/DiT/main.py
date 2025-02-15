import os.path
import random
import string
import time

import mindspore as ms
import numpy as np

from diffusion import create_diffusion
from download import load_model
from models import DiT_XL_2

num_sampling_steps = 250
cfg_scale = 4.0
# Load model:
image_size = 256
assert image_size in [256, 512], "We only provide pre-trained models for 256x256 and 512x512 resolutions."
latent_size = image_size // 8
model = DiT_XL_2(input_size=latent_size)

state_dict = load_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
ms.load_param_into_net(model, state_dict)
model.set_train(False)

# Labels to condition the model with:
class_count = 1000
l = list(range(class_count))
n = 10
class_labels_batch = [l[i:i + n] for i in range(0, len(l), n)]
repeat_num = 6  # 50 * 1000 = 50K
for i in range(repeat_num):
    print("batch %d start" % i)
    for idx, class_labels in enumerate(class_labels_batch):
        diffusion = create_diffusion(str(num_sampling_steps))
        # Create sampling noise:
        n = len(class_labels)
        stdnormal = ms.ops.StandardNormal(seed=int(time.time()))
        shape = (n, 4, latent_size, latent_size)
        z = stdnormal(shape)
        concat_op = ms.ops.Concat(axis=0)
        z = concat_op((z, z))
        y = ms.Tensor(class_labels)
        # Setup classifier-free guidance:
        concat_op = ms.ops.Concat(axis=0)
        y_null = ms.Tensor([1000] * n)
        y = concat_op((y, y_null))
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)

        # Sample images:
        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
        )
        split = ms.ops.Split(axis=0, output_num=2)
        samples, _ = split(samples)  # Remove null class samples        samples = vae.decode(samples / 0.18215).sample
        save_dir = "sample/npdata/%d/" % i
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir + '%s.npy' % (''.join(random.sample(string.ascii_letters, 8))), samples.asnumpy())
