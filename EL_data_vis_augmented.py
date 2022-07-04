

import torch                                        # root package
from torch.utils.data import DataLoader, WeightedRandomSampler    # dataset representation and loading

import torchvision

#import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import random

from training_utils import EL_dataset


#%%
# Reproducibility

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Device configuration
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ' + str(device))

num_workers = 4
pin_memory = True

#aug = config['aug']
aug_rot = True
aug_yshear = False
aug_crop = True
aug_flip = True
aug_rot90 = True
aug_blur = False
aug_sharp = False
aug_bc = True
aug_gamma = True
aug_noise = False
aug_random_erasing = False

#%%

# Load datas
all_labels = pd.read_csv('EL_data_split.csv')
train_labels = all_labels[all_labels['split']==0]
train_labels.reset_index(drop=True,inplace=True)
train_mean = train_labels['mean'].mean()
train_std = train_labels['std'].mean()
print(f'mean={train_mean:.4f}, std={train_std:.4f}')

#%%
train_dataset = EL_dataset(train_labels, im_input_size=224, train_mean=train_mean, train_std=train_std, soft_labels=False, aug_rot=False, aug_yshear=False, aug_crop=False, aug_flip=False, aug_rot90=False, aug_blur=False, aug_sharp=False, aug_bc=False, aug_gamma=False, aug_noise=False, aug_random_erasing=False)
train_dataset_aug = EL_dataset(train_labels, im_input_size=224, train_mean=train_mean, train_std=train_std, soft_labels=False, aug_rot=aug_rot, aug_yshear=aug_yshear, aug_crop=aug_crop, aug_flip=aug_flip, aug_rot90=aug_rot90, aug_blur=aug_blur, aug_sharp=aug_sharp, aug_bc=aug_bc, aug_gamma=aug_gamma, aug_noise=aug_noise, aug_random_erasing=aug_random_erasing)

N_train = len(train_labels)

# Weighted sampling
N_class = np.array([sum(train_dataset.labels==l) for l in range(5)])
w_class = 1/N_class
weights = np.array([w_class[l] for l in train_dataset.labels])
sampler = WeightedRandomSampler(weights, len(weights))    

train_loader = DataLoader(dataset=train_dataset, batch_size=8, num_workers=num_workers, sampler=sampler, pin_memory=pin_memory)
train_loader_aug = DataLoader(dataset=train_dataset_aug, batch_size=8, num_workers=num_workers, sampler=sampler, pin_memory=pin_memory)

if __name__ == '__main__':
    
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    images, labels = iter(train_loader).next()
    images *= train_std
    images += train_mean
    grid = torchvision.utils.make_grid(images)
    img = torchvision.transforms.ToPILImage()(grid)
    
    torch.manual_seed(1)
    random.seed(1)
    np.random.seed(1)
    
    images_aug, labels_aug = iter(train_loader_aug).next()
    images_aug *= train_std
    images_aug += train_mean
    grid_aug = torchvision.utils.make_grid(images_aug)
    img_aug = torchvision.transforms.ToPILImage()(grid_aug)
    
    f, (ax1, ax2) = plt.subplots(2,1, dpi=600)    
    ax1.imshow(img, vmin=0, vmax=1)
    ax1.axis('off')
    ax1.set_title('Before augmentation')
    ax2.imshow(img_aug, vmin=0, vmax=1)
    ax2.axis('off')
    ax2.set_title('After augmentation')
    f.tight_layout()
