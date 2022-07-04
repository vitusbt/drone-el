import torch 
from torch import Tensor
from torch.utils.data import Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import PIL
from PIL import Image

import numpy as np

import random

import os

from ast import literal_eval as make_tuple

#%%
class SquareTransform:
    def __init__(self, size):
        self.size = size 
    def __call__(self, im):
        s = min(im.size(-1), im.size(-2))
        return TF.resize(TF.center_crop(im, s), self.size)
    
class Rotate90:
    def __call__(self, im):
        return TF.rotate(im, 90, interpolation=transforms.InterpolationMode.BILINEAR)

class RandomGamma:
    def __init__(self, mingamma, maxgamma):
        self.minloggamma = np.log(mingamma)
        self.maxloggamma = np.log(maxgamma)
    def __call__(self, im):
        gamma = np.exp(np.random.uniform(self.minloggamma, self.maxloggamma))
        return TF.adjust_gamma(im, gamma=gamma)  
    
class AddGaussianNoise:
    def __init__(self, mean=0., minstd=0.0, maxstd=0.1):
        self.minstd = minstd
        self.maxstd = maxstd
        self.mean = mean
        
    def __call__(self, im):
        std = np.random.uniform(self.minstd, self.maxstd)
        return im + torch.randn(im.size()) * std + self.mean
    
class RandomYShearing:
    def __init__(self, degs=15):
        self.degs = degs
        
    def __call__(self, im):
        h = im.size(-2)
        w = im.size(-1)
        d = np.random.uniform(-self.degs, self.degs)
        crop_top = int((w/2)*np.tan(np.abs(d)*np.pi/180))
        crop_height = h - 2*crop_top
        y = TF.affine(im, 0, [0,0], 1, [0, d])
        return TF.resized_crop(y, crop_top, 0, crop_height, w, (h,w))
        
class RandomConstrainedCropping:
    def __init__(self, minfrac=0.7, maxfrac=1, tolerance=0.2, p=0.5):
        self.minfrac = minfrac
        self.maxfrac = maxfrac
        self.tolerance = tolerance
        self.p = p
    
    def __call__(self, im, bb):
        if random.random() < self.p:
            return im
        height, width = im.size(-2), im.size(-1)
        new_size = int(min(height, width) * random.uniform(self.minfrac, self.maxfrac))
        if bb == None:
            return transforms.RandomCrop(new_size)(im)
        box_l = bb[0]
        box_r = bb[1]
        box_t = bb[2]
        box_b = bb[3]
        box_w = box_r - box_l + 1
        box_h = box_b - box_t + 1
        
        box_l = int(box_l + box_w * self.tolerance/2)
        box_w = int(box_w*(1-self.tolerance))
        box_t = int(box_t + box_h * self.tolerance/2)
        box_h = int(box_h*(1-self.tolerance))
        box_r = box_l + box_w - 1
        box_b = box_t + box_h - 1
        
        new_size = max(new_size, box_w, box_h)
        cropped_im = im
        
        try:
            i = torch.randint(max(0, box_b - new_size), min(height - new_size, box_t) + 1, size=(1,)).item()
            j = torch.randint(max(0, box_r - new_size), min(width - new_size, box_l) + 1, size=(1,)).item()
            cropped_im = TF.crop(im, i, j, new_size, new_size)
        except:
            print(f'!!! height={height}, width={width}, box_l={bb[0]}, box_t={bb[2]}, box_r={bb[1]}, box_b={bb[3]}, new_size={new_size}')
        
        return cropped_im

def _GetTransform(im_input_size, train_mean, train_std, aug_flip=True, aug_rot90=False, aug_blur=False, aug_sharp=False, aug_bc=False, aug_gamma=False, aug_noise=False, aug_random_erasing=False):
    
    transform_list = []
    transform_list.append(SquareTransform(size=im_input_size))
    if aug_flip:
        transform_list.extend([transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5)])
    if aug_rot90:
        transform_list.append(transforms.RandomApply([Rotate90()], p=0.5))
    if aug_blur:
        transform_list.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.01,1.0))], p=0.5))
    if aug_sharp:
        transform_list.append(transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5))
    if aug_bc:
        transform_list.append(transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25)], p=0.5))
    if aug_gamma:
        transform_list.append(transforms.RandomApply([RandomGamma(2/3, 3/2)], p=0.5))
    if aug_noise:
        transform_list.append(transforms.RandomApply([AddGaussianNoise(mean=0, minstd=0.0, maxstd=0.1)], p=0.5))
    
    transform_list.append(transforms.Normalize(train_mean, train_std))
    
    if aug_random_erasing:
        transform_list.append(transforms.RandomErasing(p=0.5))
    
    return transforms.Compose(transform_list)

#%%

class EL_dataset(Dataset):
    def __init__(self, labels_data, im_input_size, train_mean, train_std, soft_labels=False, aug_rot=False, aug_yshear=False, aug_crop=False, aug_flip=True, aug_rot90=False, aug_blur=False, aug_sharp=False, aug_bc=False, aug_gamma=False, aug_noise=False, aug_random_erasing=False):
        self.labels_data = labels_data
        self.labels = self.labels_data['label']
        self.labels_multi = self.labels_data[['label_crack_a','label_crack_b','label_crack_c','label_finger_failure']]
        self.namelist = self.labels_data['name']
        self.to_tensor_transform = transforms.ToTensor()
        self.rot_transform = transforms.RandomApply([transforms.RandomRotation(degrees=2, interpolation=transforms.InterpolationMode.BILINEAR)], p=0.5) if aug_rot else None
        self.yshear_transform = transforms.RandomApply([RandomYShearing(15)], p=0.5) if aug_yshear else None
        self.crop_transform = RandomConstrainedCropping(minfrac=0.7, maxfrac=1, tolerance=0.2, p=0.5) if aug_crop else None
        self.transforms = _GetTransform(im_input_size=im_input_size,
                                        train_mean=train_mean, 
                                        train_std=train_std, 
                                        aug_flip=aug_flip,
                                        aug_rot90=aug_rot90,
                                        aug_blur=aug_blur,
                                        aug_sharp=aug_sharp, 
                                        aug_bc=aug_bc, 
                                        aug_gamma=aug_gamma, 
                                        aug_noise=aug_noise, 
                                        aug_random_erasing=aug_random_erasing)
        self.soft_labels = soft_labels
    
    def __len__(self):
        return len(self.namelist)
    
    def __getitem__(self, idx):
        ser = str(self.labels_data['series'][idx])
        img_path = os.path.join('Series'+ser, 'CellsCorr', 'Serie_'+ser+'_ImageCorr_-'+self.labels_data['name'][idx]+".png")
        image = Image.open(img_path)
        if self.soft_labels:
            label = np.zeros(5)
            label[self.labels[idx]] = 1
            label += np.concatenate((np.array([0]),self.labels_multi.loc[idx]))
            label /= label.sum()
            label = Tensor(label)
        else:
            label = self.labels[idx]
        if ser == '5':
            image = image.rotate(90, PIL.Image.NEAREST, expand = 1)
        image = self.to_tensor_transform(image)
        if self.rot_transform:
            image = self.rot_transform(image)
        if self.yshear_transform:
            # Do not shear finger failures
            if self.labels[idx] != 4:
                image = self.yshear_transform(image)
        if self.crop_transform:
            bb = make_tuple(self.labels_data['bb'][idx])
            if self.labels[idx] == 0:
                bb = None
            else:
				# Find the largest bounding box of the correct label
                bb = (min([b[0] if b[4]+1==self.labels[idx] else 10000 for b in bb]),
                      max([b[1] if b[4]+1==self.labels[idx] else 0 for b in bb]),
                      min([b[2] if b[4]+1==self.labels[idx] else 10000 for b in bb]),
                      max([b[3] if b[4]+1==self.labels[idx] else 0 for b in bb]))
                # Remember to rotate bounding box in case of series 5
                if ser == '5':
                    bb = (bb[2], bb[3], self.labels_data['width'][idx] - bb[1] - 1, self.labels_data['width'][idx] - bb[0] - 1)
                if max(bb) > 1000 or min(bb) < 0:
                    print(self.labels_data['bb'][idx] + ", lab=" + str(self.labels[idx]) + ", " + str(bb) + ", idx="+str(idx))
            image = self.crop_transform(image, bb)
        # Do all the standard transforms
        image = self.transforms(image)
        return image, label



#%%

class Model:
    def __init__(self, net, loss_fn, N_classes, N_train, N_test, num_epochs=200, soft_labels=False, lr=0.001, wd=0, scheduler_gamma=0.99):
        self.net = net
        self.optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=scheduler_gamma)
        self.loss_fn = loss_fn
        self.N_classes = N_classes
        self.N_train = N_train
        self.N_test = N_test
        
        self.running_train_loss = 0
        self.running_test_loss = 0
        self.train_loss_vals = np.empty(num_epochs)
        self.test_loss_vals = np.empty(num_epochs)
        
        self.train_outputs = np.empty((num_epochs, self.N_train, self.N_classes))
        self.train_labels = np.empty((num_epochs, self.N_train, self.N_classes)) if soft_labels else np.empty((num_epochs, self.N_train))
        
        self.test_outputs = np.empty((num_epochs, self.N_test, self.N_classes))
        self.test_labels = np.empty((num_epochs, self.N_test, self.N_classes)) if soft_labels else np.empty((num_epochs, self.N_test))
        
        self.test_counter = 0
        self.train_counter = 0
    
    # Take one optimization step on a training batch
    def step(self, epoch, images, labels):
        outputs = self.net(images)
        loss = self.loss_fn(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        n_batch = labels.size(0)
        self.running_train_loss += loss.item()*n_batch
        self.train_outputs[epoch, self.train_counter:(self.train_counter+n_batch)] = outputs.cpu().detach().numpy()
        self.train_labels[epoch, self.train_counter:(self.train_counter+n_batch)] = labels.cpu().detach().numpy()
        self.train_counter += n_batch
        
    # Evaluate and save outputs on test batch
    def eval_test(self, epoch, images, labels):
        with torch.no_grad():
            outputs = self.net(images)
            loss = self.loss_fn(outputs, labels)
            n_batch = labels.size(0)
            self.running_test_loss += loss.item()*n_batch
            self.test_outputs[epoch, self.test_counter:(self.test_counter+n_batch)] = outputs.cpu().detach().numpy()
            self.test_labels[epoch, self.test_counter:(self.test_counter+n_batch)] = labels.cpu().detach().numpy()
            self.test_counter += n_batch
            
    # Switch between training and evaluation (test) mode
    def set_eval_mode(self, mode=True):
        if mode:
            self.net.eval()
        else:
            self.net.train()
    
    # Save the accumulated test loss at the end of an epoch
    def save_loss(self, epoch):
        self.running_train_loss /= self.N_train
        self.train_loss_vals[epoch] = self.running_train_loss
        self.running_train_loss = 0
        self.train_counter = 0
        self.scheduler.step()
        
    # Calculate and save metrics on both the train and test sets
    def save_metrics(self, epoch):
        self.running_test_loss /= self.N_test
        self.test_loss_vals[epoch] = self.running_test_loss
        self.running_test_loss = 0
        self.test_counter = 0
        