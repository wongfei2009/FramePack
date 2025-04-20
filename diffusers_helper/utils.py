import os
import cv2
import json
import random
import glob
import torch
import einops
import numpy as np
import datetime
import torchvision

from PIL import Image


def resize_and_center_crop(image, target_width, target_height):
    if target_height == image.shape[0] and target_width == image.shape[1]:
        return image

    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(target_width / original_width, target_height / original_height)
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize((resized_width, resized_height), Image.LANCZOS)
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def batch_mixture(a, b=None, probability_a=0.5, mask_a=None):
    batch_size = a.size(0)

    if b is None:
        b = torch.zeros_like(a)

    if mask_a is None:
        mask_a = torch.rand(batch_size) < probability_a

    mask_a = mask_a.to(a.device)
    mask_a = mask_a.reshape((batch_size,) + (1,) * (a.dim() - 1))
    result = torch.where(mask_a, a, b)
    return result


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def supress_lower_channels(m, k, alpha=0.01):
    data = m.weight.data.clone()

    assert int(data.shape[1]) >= k

    data[:, :k] = data[:, :k] * alpha
    m.weight.data = data.contiguous().clone()
    return m


def soft_append_bcthw(history, current, overlap=0):
    if overlap <= 0:
        return torch.cat([history, current], dim=2)

    assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
    assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"
    
    weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device).view(1, 1, -1, 1, 1)
    blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
    output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

    return output.to(history)


def save_bcthw_as_mp4(x, output_filename, fps=10, crf=0):
    b, c, t, h, w = x.shape

    per_row = b
    for p in [6, 5, 4, 3, 2]:
        if b % p == 0:
            per_row = p
            break

    os.makedirs(os.path.dirname(os.path.abspath(os.path.realpath(output_filename))), exist_ok=True)
    x = torch.clamp(x.float(), -1., 1.) * 127.5 + 127.5
    x = x.detach().cpu().to(torch.uint8)
    x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=per_row)
    torchvision.io.write_video(output_filename, x, fps=fps, video_codec='libx264', options={'crf': str(int(crf))})
    return x


def pytorch2numpy(imgs):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)
        y = y * 127.5 + 127.5
        y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        results.append(y)
    return results


def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.5 - 1.0
    h = h.movedim(-1, 1)
    return h


def duplicate_prefix_to_suffix(x, count, zero_out=False):
    if zero_out:
        return torch.cat([x, torch.zeros_like(x[:count])], dim=0)
    else:
        return torch.cat([x, x[:count]], dim=0)


def expand_to_dims(x, target_dims):
    return x.view(*x.shape, *([1] * max(0, target_dims - x.dim())))


def repeat_to_batch_size(tensor: torch.Tensor, batch_size: int):
    if tensor is None:
        return None

    first_dim = tensor.shape[0]

    if first_dim == batch_size:
        return tensor

    if batch_size % first_dim != 0:
        raise ValueError(f"Cannot evenly repeat first dim {first_dim} to match batch_size {batch_size}.")

    repeat_times = batch_size // first_dim

    return tensor.repeat(repeat_times, *[1] * (tensor.dim() - 1))


def crop_or_pad_yield_mask(x, length):
    B, F, C = x.shape
    device = x.device
    dtype = x.dtype

    if F < length:
        y = torch.zeros((B, length, C), dtype=dtype, device=device)
        mask = torch.zeros((B, length), dtype=torch.bool, device=device)
        y[:, :F, :] = x
        mask[:, :F] = True
        return y, mask

    return x[:, :length, :], torch.ones((B, length), dtype=torch.bool, device=device)


def generate_timestamp():
    now = datetime.datetime.now()
    timestamp = now.strftime('%y%m%d_%H%M%S')
    milliseconds = f"{int(now.microsecond / 1000):03d}"
    random_number = random.randint(0, 9999)
    return f"{timestamp}_{milliseconds}_{random_number}"






