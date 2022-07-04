import argparse
 
parser = argparse.ArgumentParser(description="EL training",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-e", "--num_epochs", type=int, default=300, help="number of epochs")
parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size")
parser.add_argument("-l", "--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("-g", "--scheduler_gamma", type=float, default=0.99, help="gamma value for exp decay learning rate scheduler")
parser.add_argument("-w", "--weight_decay", type=float, default=0, help="weight decay (L2 regularization)")
parser.add_argument("-s", "--label_smoothing", type=float, default=0, help="label smoothing")
parser.add_argument("-S", "--soft_labels", action='store_true', help="use soft labels (for multilabeled data)")
parser.add_argument("-u", "--unfreeze", type=int, default=0, help="unfreeze weights in conv layers after this many epochs (sequential training)")

parser.add_argument("-k", "--kfold_calib", type=int, default=-1, help="which fold to use as calibration set")
parser.add_argument("-t", "--test", action='store_true', help="use outer split for testing")

#parser.add_argument("--aug", action='store_true', help="aug")
parser.add_argument("--aug_rot", action='store_true', help="random rotations up to 1 deg")
parser.add_argument("--aug_yshear", action='store_true', help="random y shearing")
parser.add_argument("--aug_crop", action='store_true', help="random cropping")
parser.add_argument("--aug_flip", action='store_false', help="DISABLE random flipping")
parser.add_argument("--aug_rot90", action='store_true', help="random 90 deg rotation")
parser.add_argument("--aug_blur", action='store_true', help="random gaussian blurring")
parser.add_argument("--aug_sharp", action='store_true', help="random sharpness adj")
parser.add_argument("--aug_bc", action='store_true', help="random brightness/contrast adj")
parser.add_argument("--aug_gamma", action='store_true', help="random gamma adj")
parser.add_argument("--aug_noise", action='store_true', help="random gaussian noise")
parser.add_argument("--aug_random_erasing", action='store_true', help="random erasing")

parser.add_argument("-D", "--deterministic", action='store_true', help="deterministic (slower)")
parser.add_argument("-W", "--num_workers", type=int, default=2, help="number of workers in data loader")
parser.add_argument("-P", "--pin_memory", action='store_true', help="pin memory in data loader")
parser.add_argument("-E", "--save_model_epochs", type=int, default=0, help="save the model every this number of epochs")
parser.add_argument("-O", "--save_optim", action='store_true', help="save the optimizer state at the end")
parser.add_argument("--seed", type=int, default=0, help="random seed")

parser.add_argument("name", help="Name to be saved")
parser.add_argument("net", help="Net to load")
args = parser.parse_args()
config = vars(args)
print(config)

import torch  
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler 

import numpy as np
import pandas as pd

import random
import time

from nets import CreateNet
from training_utils import EL_dataset, Model


#%%
# Reproducibility
seed = config['seed']
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
if config['deterministic']:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Device configuration
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ' + str(device))

num_workers = config['num_workers']
pin_memory = config['pin_memory']

# Hyper-parameters 
num_epochs = config['num_epochs']
batch_size = config['batch_size']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
label_smoothing = config['label_smoothing']
scheduler_gamma = config['scheduler_gamma']
soft_labels = config['soft_labels']
unfreeze = config['unfreeze']
kfold_calib = config['kfold_calib']
test = config['test']

#aug = config['aug']
aug_rot = config['aug_rot']
aug_yshear = config['aug_yshear']
aug_crop = config['aug_crop']
aug_flip = config['aug_flip']
aug_rot90 = config['aug_rot90']
aug_blur = config['aug_blur']
aug_sharp = config['aug_sharp']
aug_bc = config['aug_bc']
aug_gamma = config['aug_gamma']
aug_noise = config['aug_noise']
aug_random_erasing = config['aug_random_erasing']

model_name = config['name']
net, im_input_size = CreateNet(config['net'], unfreeze>0)

net = net.to(device)
if seed != 0:
    torch.manual_seed(seed)

save_model_epochs = config['save_model_epochs']
save_optim = config['save_optim']

#%%

# Load datas
all_labels = pd.read_csv('EL_data_split.csv')

# Load dataset and split into training and validation (and calibration) sets
train_labels = None
val_labels = None
calib_labels = None
if test:
    val_labels = all_labels[all_labels['split']==2]
    if kfold_calib == -1:
        train_labels = all_labels[all_labels['split']<=1]
    else:
        train_labels = all_labels[(all_labels['kfold_outer']!=kfold_calib) & (all_labels['kfold_outer']!=-1)]
        calib_labels = all_labels[all_labels['kfold_outer']==kfold_calib]
else:
    val_labels = all_labels[all_labels['split']==1]
    if kfold_calib == -1:
        train_labels = all_labels[all_labels['split']==0]
    else:
        train_labels = all_labels[(all_labels['kfold_inner']!=kfold_calib) & (all_labels['kfold_inner']!=-1)]
        calib_labels = all_labels[all_labels['kfold_inner']==kfold_calib]
train_labels.reset_index(drop=True,inplace=True)
val_labels.reset_index(drop=True,inplace=True)
if kfold_calib >= 0:
    calib_labels.reset_index(drop=True,inplace=True)

train_mean = train_labels['mean'].mean()
train_std = train_labels['std'].mean()
print(f'mean={train_mean:.4f}, std={train_std:.4f}')

#%%
train_dataset = EL_dataset(train_labels, im_input_size=im_input_size, train_mean=train_mean, train_std=train_std, soft_labels=soft_labels, aug_rot=aug_rot, aug_yshear=aug_yshear, aug_crop=aug_crop, aug_flip=aug_flip, aug_rot90=aug_rot90, aug_blur=aug_blur, aug_sharp=aug_sharp, aug_bc=aug_bc, aug_gamma=aug_gamma, aug_noise=aug_noise, aug_random_erasing=aug_random_erasing)
val_dataset = EL_dataset(val_labels, im_input_size=im_input_size, train_mean=train_mean, train_std=train_std, soft_labels=soft_labels)
calib_dataset = EL_dataset(calib_labels, im_input_size=im_input_size, train_mean=train_mean, train_std=train_std, soft_labels=soft_labels) if kfold_calib >= 0 else None

N_train = len(train_labels)
N_val = len(val_labels)
N_calib = len(calib_labels) if kfold_calib >= 0 else 0

# Weighted sampling
N_class = np.array([sum(train_dataset.labels==l) for l in range(5)])
w_class = 1/N_class
weights = np.array([w_class[l] for l in train_dataset.labels])
sampler = WeightedRandomSampler(weights, len(weights))    

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler, pin_memory=pin_memory)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
calib_loader = DataLoader(dataset=calib_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) if kfold_calib >= 0 else None

#%%
if __name__=='__main__':
    
    model = Model(net, nn.CrossEntropyLoss(label_smoothing=label_smoothing), 5, N_train, N_val+N_calib, num_epochs=num_epochs, soft_labels=soft_labels, lr=learning_rate, wd=weight_decay, scheduler_gamma=scheduler_gamma)
    
    n_total_steps = len(train_loader)
    
    print('Starting training')
    start_time = time.perf_counter()
    
    for epoch in range(num_epochs):
		
		# Unfreeze weights
        if ((epoch == unfreeze) and (unfreeze > 0)):
            for param in model.net.features.parameters():
                param.requires_grad = True
        
        model.set_eval_mode(False)
        #print(f'EPOCH {epoch+1}')
        
        for step, (images, labels) in enumerate(train_loader):
            if not soft_labels:
                print(str(labels[0].item()), end="")
            images = images.to(device)
            labels = labels.to(device)

            model.step(epoch, images, labels)
            
        if save_model_epochs > 0:
            if (epoch+1) % save_model_epochs == 0:
                torch.save(model.net.state_dict(), model_name+'_epoch'+str(epoch+1)+'.pt')
        
        model.set_eval_mode(True)
        #print('Evaluating validation images')
        
        for num, (val_images, val_labels) in enumerate(val_loader):
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
            model.eval_test(epoch, val_images, val_labels)    
        
        if kfold_calib >= 0:
            for num, (calib_images, calib_labels) in enumerate(calib_loader):
                calib_images = calib_images.to(device)
                calib_labels = calib_labels.to(device)
                model.eval_test(epoch, calib_images, calib_labels)      
        model.save_loss(epoch)
        model.save_metrics(epoch)
        
        print(f'Epoch {epoch+1}, TrainLoss={model.train_loss_vals[epoch]:.4e}, ValLoss={model.test_loss_vals[epoch]:.4e}, LR={model.scheduler.get_last_lr()[0]:.4e}, Time={(time.perf_counter()-start_time):.2f}')
        
    
    print(f'Finished training, Time={(time.perf_counter()-start_time):.2f}')

    if save_model_epochs > 0:
        if num_epochs % save_model_epochs != 0:
            torch.save(model.net.state_dict(), model_name+'_epoch'+str(epoch+1)+'.pt')
    if save_optim:
        torch.save(model.optimizer.state_dict(), model_name+'_optim'+str(epoch+1)+'.pt')
    np.savez_compressed(model_name+'_outputs.npz', 
             train_loss = model.train_loss_vals, 
             test_loss = model.test_loss_vals,
             train_outputs = model.train_outputs,
             test_outputs = model.test_outputs,
             train_labels = model.train_labels,
             test_labels = model.test_labels,
            )
    
    print('Saved outputs')

