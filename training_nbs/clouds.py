from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T

from fastprogress import progress_bar

def ls(path: Path): 
    return sorted(list(Path(path).iterdir()))

class CloudDataset:
    
    def __init__(self, 
                 files, # list of numpy files to load (they come from the artifact)
                 num_frames=4, # how many consecutive frames to stack
                 scale=True, # if we images to interval [-0.5, 0.5]
                 size=64, # resize dim, original images are big (446, 780)
                 valid=False, # if True, transforms are deterministic
                ):
        
        tfms = [T.Resize((size, int(size*1.7)))] if size is not None else []
        tfms += [T.RandomCrop(size)] if not valid else [T.CenterCrop(size)]
        self.tfms = T.Compose(tfms)
        
        
        data = []
        for file in progress_bar(files, leave=False):
            one_day = np.load(file)
            if scale:
                one_day = 0.5 - self._scale(one_day)
        
            wds = np.lib.stride_tricks.sliding_window_view(
                one_day.squeeze(), 
                num_frames, 
                axis=0).transpose((0,3,1,2))
            data.append(wds)
        self.data = np.concatenate(data, axis=0)
            
    @staticmethod
    def _scale(arr):
        "Scales values of array in [0,1]"
        m, M = arr.min(), arr.max()
        return (arr - m) / (M - m)
    
    def __getitem__(self, idx):
        return self.tfms(torch.from_numpy(self.data[idx]))
    
    def __len__(self): return len(self.data)

    def save(self, fname="cloud_frames.npy"):
        np.save(fname, self.data)
        

        
if __name__=="__main__":
    ## Get dataset
    ## It's one file per day

    import wandb
    with wandb.init(project="ddpm_clouds"):
        artifact = wandb.use_artifact('capecape/gtc/np_dataset:v0', type='dataset')
        artifact_dir = artifact.download()

    files = ls(Path(artifact_dir))

    train_ds = CloudDataset(files)

    print(train_ds[0:5].shape)