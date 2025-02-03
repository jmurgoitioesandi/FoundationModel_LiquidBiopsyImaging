## Standard libraries
import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from SimCLR import SimCLR
from datasets import loading_datasets

from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', 'path_to_your_logs'])
url = tb.launch()
print(f"TensorBoard is running at {url}")

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "/project/aoberai_286/RCD/FoundationModel_Liquidbiopsy/checkpoints"
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count() // 8
## Path to the file with the rare tiles
RAREST_PATH = 'rare_tiles.npy'
## Path to the slides
SLIDES_PATH = '/scratch1/murgoiti/latestageBC_slides'
## Load list of slides to consider
with open(f'/project/aoberai_286/RCD/FoundationModel_Liquidbiopsy/slide_list.txt') as f:
    SLIDE_LIST = f.read().split('\n')


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, 'SimCLR'),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5', every_n_epochs=5),
                                    LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = None # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        dataset_normal, dataset_rare, dataset_normal_val, dataset_rare_val = loading_datasets(RAREST_PATH, SLIDES_PATH, SLIDE_LIST)

        # train_loader_rare = data.DataLoader(dataset_rare, batch_size=batch_size, shuffle=True,
        #                                drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        train_loader_normal = data.DataLoader(dataset_normal, batch_size=batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        # val_loader_rare = data.DataLoader(dataset_rare_val, batch_size=batch_size, shuffle=False,
        #                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader_normal = data.DataLoader(dataset_normal_val, batch_size=batch_size, shuffle=False,
                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
        # train_loader = zip(train_loader_rare, train_loader_normal)
        # val_loader = zip(val_loader_rare, val_loader_normal)
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader_normal, val_loader_normal)
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    return model

simclr_model = train_simclr(batch_size=500,
                            hidden_dim=128,
                            lr=5e-4,
                            temperature=0.07,
                            weight_decay=1e-4,
                            max_epochs=1000)
