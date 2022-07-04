import PIL
from PIL import Image, ImageEnhance

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import random

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import scipy.io

from sklearn.decomposition import PCA

#%%
def LoadImage(dataset, idx, rotate=True, GS=False):
    t = 'GS' if GS else 'Corr'
    ser = str(dataset['series'][idx])
    img_path = os.path.join('Series'+ser, 'Cells'+t, 'Serie_'+ser+'_Image'+t+'_-'+dataset['name'][idx]+'.png')
    image = Image.open(img_path)
    if rotate and ser == '5':
        image = image.rotate(90, PIL.Image.NEAREST, expand = 1)
    return image

def LoadMask(dataset, idx, rotate=True):
    ser = str(dataset['series'][idx])
    mask_path = os.path.join('Series'+ser, 'MaskGT', 'GT_Serie_'+ser+'_Image_-'+dataset['name'][idx]+'.mat')
    if not os.path.exists(mask_path):
        return None
    mat = scipy.io.loadmat(mask_path)
    GTLabel = mat["GTLabel"]
    f_labels = [item[0][0] for item in GTLabel]
    GTMask = mat["GTMask"]
    if len(GTMask.shape)==2:
        GTMask = GTMask.reshape((GTMask.shape[0], GTMask.shape[1], 1))
    if rotate and ser == '5':
        GTMask = np.rot90(GTMask)
    return (f_labels, GTMask)

def cmap(x):
    return {
        'Crack A': mpl.colors.ListedColormap(['#00ff00']),
        'Crack B': mpl.colors.ListedColormap(['#ffff00']),
        'Crack C': mpl.colors.ListedColormap(['#ff4040']),
        'Finger Failure': mpl.colors.ListedColormap(['#0040ff'])
    }[x]

def ShowImages(dataset, ids, masked=True, title=True):
    n = len(ids)
    imgs = [LoadImage(dataset, ids[i]) for i in range(n)]
    if masked:
        masks = [LoadMask(dataset, ids[i]) for i in range(n)]
        for i in range(n):
            plt.figure(figsize=(8,4))
            if title:
                plt.suptitle(str(tuple(np.round(masks[i][2], 5)))+'\n['+str(ids[i])+'] Ser'+str(dataset['series'][ids[i]])+' '+dataset['name'][ids[i]])
            plt.subplot(1,2,1)
            plt.imshow(imgs[i])
            plt.subplot(1,2,2)
            plt.imshow(imgs[i])
            for j in range(len(masks[i][0])):
                mask = masks[i][1][:,:,j]
                masked = np.ma.masked_where(mask == 0, mask)
                #ax2.imshow(masked, cmap(masks[i][0][j]), alpha=0.75)
                plt.imshow(masked, cmap(masks[i][0][j]), alpha=0.75)
            plt.show()
    else:
        for i in range(n):
            plt.figure(figsize=(4,4))
            plt.imshow(imgs[i])
            #plt.title('No defect')
            plt.show()
            
def ShowRandomImages(dataset, n, masked=True, title=True):
    ids = np.random.choice(len(dataset), size=n, replace=False)
    ShowImages(dataset, ids, masked, title)
    
def ShowSortedImages(dataset, n, col, masked=True, title=True, decr=False):
    x = dataset[col]
    if decr:
        x = -x
    ids = np.argpartition(x, n)[:n]
    ids = ids[np.argsort(x[ids])].reset_index(drop=True)
    print(str(ids))
    print(x[ids])
    ShowImages(dataset, ids, masked, title)



def LoadImages(dataset, n=None, ids=None, GS=False):
    if ids is None:
        ids = np.random.choice(len(dataset), size=n, replace=False)
        print(ids)
    imgs = [LoadImage(dataset, i, GS=GS) for i in ids]
    masks = [LoadMask(dataset, i) for i in ids]
    return (imgs, masks)

def ShowImagesGrid(imgs, masks=None, rows=2, cols=6, figsize=None):
    if figsize == None:
        figsize = (2*cols, 2*rows)
    plt.gray()
    fig, axs = plt.subplots(rows, cols, dpi=600, figsize=figsize)
    n = len(imgs)
    #fig.suptitle('Horizontally stacked subplots')
    for i in range(n):
        axs.flat[i].set_axis_off()
        axs.flat[i].imshow(imgs[i], vmin=0, vmax=255)
    
    if masks:
        for i in range(n):
            axs.flat[i+n].set_axis_off()
            enhancer = ImageEnhance.Brightness(imgs[i])
            axs.flat[i+n].imshow(enhancer.enhance(0.2), vmin=0, vmax=255)
            for j in range(len(masks[i][0])):
                mask = masks[i][1][:,:,j]
                masked = np.ma.masked_where(mask == 0, mask)
                axs.flat[i+n].imshow(masked, cmap(masks[i][0][j]), alpha=0.75)

    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.03, hspace=0.03)
    
    plt.show()

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def concat_masks_h(m1, m2):
    new_mask = np.concatenate((m1[1], m2[1]), axis=1)
    return (m1[0], new_mask)

def concat_masks_v(m1, m2):
    new_mask = np.concatenate((m1[1], m2[1]), axis=0)
    return (m1[0], new_mask)

#%%

if __name__ == '__main__':
    
    # Load data
    df = pd.read_csv('EL_data.csv')
    df_filt = pd.read_csv('EL_data_split.csv')

    df_filt_no_defect = df_filt[(df_filt['npixels_crack_a']==0)
                                &(df_filt['npixels_crack_b']==0)
                                &(df_filt['npixels_crack_c']==0)
                                &(df_filt['npixels_finger_failure']==0)]

    df_filt_crack_a = df_filt[(df_filt['label_crack_a']==1)
                                &(df_filt['npixels_crack_b']==0)
                                &(df_filt['npixels_crack_c']==0)
                                &(df_filt['npixels_finger_failure']==0)]

    df_filt_crack_b = df_filt[(df_filt['npixels_crack_a']==0)
                                &(df_filt['label_crack_b']==1)
                                &(df_filt['npixels_crack_c']==0)
                                &(df_filt['npixels_finger_failure']==0)]

    df_filt_crack_c = df_filt[(df_filt['npixels_crack_a']==0)
                                &(df_filt['npixels_crack_b']==0)
                                &(df_filt['label_crack_c']==1)
                                &(df_filt['npixels_finger_failure']==0)]

    df_filt_finger_failure = df_filt[(df_filt['npixels_crack_a']==0)
                                &(df_filt['npixels_crack_b']==0)
                                &(df_filt['npixels_crack_c']==0)
                                &(df_filt['label_finger_failure']==1)]

    df_filt_multi = df_filt[df_filt['label_crack_a']
                            +df_filt['label_crack_b']
                            +df_filt['label_crack_c']
                            +df_filt['label_finger_failure'] >= 2]

    df_filt_label_overlap = df_filt[(df_filt['label_crack_a']
                            +df_filt['label_crack_b']
                            +df_filt['label_crack_c']
                            +df_filt['label_finger_failure'] == 0)
                            &(df_filt['npixels_crack_a']
                            +df_filt['npixels_crack_b']
                            +df_filt['npixels_crack_c']
                            +df_filt['npixels_finger_failure'] >= 2)]

    df_dark = df[df['GS_mean']<0.30]

    df_filt_no_defect.reset_index(drop=True,inplace=True)
    df_filt_crack_a.reset_index(drop=True,inplace=True)
    df_filt_crack_b.reset_index(drop=True,inplace=True)
    df_filt_crack_c.reset_index(drop=True,inplace=True)
    df_filt_finger_failure.reset_index(drop=True,inplace=True)
    df_filt_multi.reset_index(drop=True,inplace=True)
    df_filt_label_overlap.reset_index(drop=True,inplace=True)
    df_dark.reset_index(drop=True,inplace=True)


    summary = pd.DataFrame([[sum((df_filt['label']==i) & (df_filt['series']==s)) for i in range(5)] for s in range(1,7)])

    
    
    random.seed(1)
    np.random.seed(1)
    
    plt.gray()
    
    summary = np.array([[((df_filt['split']==i) & (df_filt['label']==c)).sum() for c in range(5)] for i in range(3)])
    
    # Show some no defect images
    imgs, _ = LoadImages(df_filt_no_defect, n=6)
    ShowImagesGrid(imgs, rows=1, cols=6)
    
    # Show some images from each class
    #ids = np.array([81, 48, 72, 108, 39])
    imgs, masks = LoadImages(df_filt_crack_a, n=3)
    ShowImagesGrid(imgs, masks, rows=2, cols=3)
    
    imgs, masks = LoadImages(df_filt_crack_b, n=3)
    ShowImagesGrid(imgs, masks, rows=2, cols=3)
    
    imgs, masks = LoadImages(df_filt_crack_c, n=3)
    ShowImagesGrid(imgs, masks, rows=2, cols=3)
    
    imgs, masks = LoadImages(df_filt_finger_failure, n=3)
    ShowImagesGrid(imgs, masks, rows=2, cols=3)
    
    # Show some images with multiple labels
    imgs, masks = LoadImages(df_filt_multi, n=6)
    ShowImagesGrid(imgs, masks, rows=2, cols=6)
    
    
    # Show some images where the label overlaps 
    imgs, masks = LoadImages(df_filt, ids=[1028,1029])
    concat_img = get_concat_h(imgs[0], imgs[1])
    concat_mask = concat_masks_h(masks[0], masks[1])
    ShowImagesGrid([concat_img], [concat_mask], rows=2, cols=1, figsize=(4,4))
    
    imgs, masks = LoadImages(df_filt, ids=[16931,16937])
    concat_img = get_concat_v(imgs[0], imgs[1])
    concat_mask = concat_masks_v(masks[0], masks[1])
    ShowImagesGrid([concat_img], [concat_mask], rows=1, cols=2, figsize=(4,4))
    
    
    # Show some dark images
    imgs, _ = LoadImages(df_dark, ids=[4, 68], GS=True)
    ShowImagesGrid(imgs, rows=1, cols=2)
    
    
    # Show PCA
    
    X = df[['GS_mean','GS_std','GS_skew']]
    X_scaled = X.copy()
    X_scaled = X_scaled - np.mean(X_scaled)
    X_scaled = X_scaled/np.std(X_scaled)

    col = ['red' if m<0.3 else 'black' for m in X['GS_mean']]
    lab = ['mean < 0.3' if m<0.3 else 'mean > 0.3' for m in X['GS_mean']]

    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    X_scaled_PCA = pca.fit_transform(X_scaled).transpose()
    
    plt.figure(figsize=(6,4.5), dpi=600)
    plt.scatter(X_scaled_PCA[0][X['GS_mean'] >= 0.30], X_scaled_PCA[1][X['GS_mean'] >= 0.30], s=1.5, c='black', label='mean \u2265 0.3', alpha=0.25)
    plt.scatter(X_scaled_PCA[0][X['GS_mean'] < 0.30], X_scaled_PCA[1][X['GS_mean'] < 0.30], s=1.5, c='red', label='mean < 0.3', alpha=0.5)
    qlow = np.quantile(X_scaled_PCA, 0.01, axis=1)
    qhigh = np.quantile(X_scaled_PCA, 0.99, axis=1)
    llow = np.max([np.min(X_scaled_PCA, axis=1), qlow - (qhigh-qlow)*1], axis=0)
    lhigh = np.min([np.max(X_scaled_PCA, axis=1), qhigh + (qhigh-qlow)*1], axis=0)
    plt.xlim(llow[0],lhigh[0])
    plt.ylim(llow[1],lhigh[1])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.show()
    