from download import convert_model

image_size = 256
assert image_size in [256, 512]

convert_model(f"DiT-XL-2-{image_size}x{image_size}.pt")
