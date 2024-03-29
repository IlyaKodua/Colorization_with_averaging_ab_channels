
import glob
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from utils import*
from model import*
Path.ls = lambda x: list(x.iterdir())

import torch


from sig import*
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from dataset import*

paths = glob.glob("/home/liya/devel/to_png/img_jpg/*.png") # Your path for your dataset
np.random.seed(123)
paths_subset = np.random.choice(paths, 20, replace=False) # choosing 1000 images randomly
rand_idxs = np.random.permutation(20)
train_idxs = rand_idxs[:20] # choosing the first 8000 as training set
val_idxs = rand_idxs[20:] # choosing last 2000 as validation set
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))


_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")


train_dl = make_dataloaders(batch_size=8, paths=train_paths, split='train')
val_dl = make_dataloaders(batch_size=8, paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))





def train_model(model, train_dl, epochs, display_every=1):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for i,data in enumerate(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs
            if i % 1000 == 3:
                print(1)
                # model.save()

gen = siggraph17(False)
model = MainModel(gen)
# model.load()
train_model(model, train_dl, 10)
model.save()








