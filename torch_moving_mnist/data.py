# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_data.ipynb.

# %% auto 0
__all__ = ['mnist_stats', 'affine_params', 'padding', 'apply_n_times', 'RandomTrajectory', 'MovingMNIST']

# %% ../nbs/01_data.ipynb 2
from functools import partial
from types import SimpleNamespace
from torchvision.datasets import MNIST

# %% ../nbs/01_data.ipynb 10
import random

import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF

mnist_stats    = ([0.131], [0.308])

# %% ../nbs/01_data.ipynb 17
def padding(img_size=64, mnist_size=28): return (img_size - mnist_size) // 2

# %% ../nbs/01_data.ipynb 22
def apply_n_times(tf, x, n=1):
    "Apply `tf` to `x` `n` times, return all values"
    sequence = [x]
    for n in range(n):
        sequence.append(tf(sequence[n]))
    return sequence

show_images(apply_n_times(tf, pdigit, n=5), figsize=(10,20))

# %% ../nbs/01_data.ipynb 24
affine_params = SimpleNamespace(
    angle=(-4, 4),
    translate=((-5, 5), (-5, 5)),
    scale=(.8, 1.2),
    shear=(-3, 3),
)

# %% ../nbs/01_data.ipynb 27
class RandomTrajectory:
    def __init__(self, affine_params, n=5, **kwargs):
        self.angle     = random.uniform(*affine_params.angle)
        self.translate = (random.uniform(*affine_params.translate[0]), 
                          random.uniform(*affine_params.translate[1]))
        self.scale     = random.uniform(*affine_params.scale)
        self.shear     = random.uniform(*affine_params.shear)
        self.n = n
        self.tf = partial(TF.affine, angle=self.angle, translate=self.translate, scale=self.scale, shear=self.shear, **kwargs)
            
    def __call__(self, img):
        return apply_n_times(self.tf, img, n=self.n)
    
    def __repr__(self):
        s = ("RandomTrajectory(\n"
             f"  angle:     {self.angle}\n"
             f"  translate: {self.translate}\n"
             f"  scale:     {self.scale}\n"
             f"  shear:     {self.shear}\n)")
        return s

# %% ../nbs/01_data.ipynb 33
import math
import random
from fastprogress import progress_bar

class MovingMNIST:
    def __init__(self, path=".",  # path to store the MNIST dataset
                 affine_params: dict=affine_params, # affine transform parameters, refer to torchvision.transforms.functional.affine
                 num_digits: list[int]=[1,2], # how many digits to move, random choice between the value provided
                 num_frames: int=4, # how many frames to create
                 img_size=64, # the canvas size, the actual digits are always 28x28
                 concat=True, # if we concat the final results (frames, 1, 28, 28) or a list of frames.
                 normalize=False # scale images in [0,1] and normalize them with MNIST stats. Applied at batch level. Have to take care of the canvas size that messes up the stats!
                ):
        self.mnist = MNIST(path, download=True).data
        self.affine_params = affine_params
        self.num_digits = num_digits
        self.num_frames = num_frames
        self.img_size = img_size
        self.pad = padding(img_size)
        self.concat = concat
        
        # some computation to ensure normalizing correctly-ish
        batch_tfms = [T.ConvertImageDtype(torch.float32)]
        if normalize:
            ratio = (28/img_size)**2*max(num_digits)
            mean, std = mnist_stats
            scaled_mnist_stats = ([mean[0]*ratio], [std[0]*ratio])
            print(f"New computed stats for MovingMNIST: {scaled_mnist_stats}")
            batch_tfms += [T.Normalize(*scaled_mnist_stats)] if normalize else []
        self.batch_tfms = T.Compose(batch_tfms)  
    
    def random_place(self, img):
        "Randomly place the digit inside the canvas"
        x = random.uniform(-self.pad, self.pad)
        y = random.uniform(-self.pad, self.pad)
        return TF.affine(img, translate=(x,y), angle=0, scale=1, shear=(0,0))
    
    def random_digit(self):
        "Get a random MNIST digit randomly placed on the canvas"
        img = self.mnist[[random.randrange(0, len(self.mnist))]]
        pimg = TF.pad(img, padding=self.pad)
        return self.random_place(pimg)
    
    def _one_moving_digit(self):
        digit = self.random_digit()
        traj = RandomTrajectory(self.affine_params, n=self.num_frames-1)
        return torch.stack(traj(digit))
    
    def __getitem__(self, i):
        moving_digits = [self._one_moving_digit() for _ in range(random.choice(self.num_digits))]
        moving_digits = torch.stack(moving_digits)
        combined_digits = moving_digits.max(dim=0)[0]
        return combined_digits if self.concat else [t.squeeze(dim=0) for t in combined_digits.split(1)]
    
    def get_batch(self, bs=32):
        "Grab a batch of data"
        batch = torch.stack([self[0] for _ in range(bs)])
        return self.batch_tfms(batch) if self.batch_tfms is not None else batch
    
    def save(self, fname="mmnist.pt", n_batches=2, bs=32):
        data = [] 
        for _ in progress_bar(range(n_batches)):
            data.append(self.get_batch(bs=bs))
        
        data = torch.cat(data, dim=0)
        print("Saving dataset")
        torch.save(data, f"{fname}")
