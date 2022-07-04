import numpy as np
import pandas as pd

import scipy.io
from scipy.stats import skew, kurtosis

from PIL import Image

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import random
from sklearn.model_selection import train_test_split, KFold

failure_types = ['Crack A', 'Crack B', 'Crack C', 'Finger Failure']
trim_px = 10 # How much to trim off each side of the mask

#%%

def ReadMaskData(file):
    
    if not os.path.exists(file):
        return ([0]*4, [0]*4, 0, 0, 0, 0)
    
    mat = scipy.io.loadmat(file)
    
    GTLabel = mat['GTLabel']
    layer_labels = np.array([item[0][0] for item in GTLabel])
    
    GTMask = mat['GTMask']
    if len(GTMask.shape)==2:
        GTMask = GTMask.reshape((GTMask.shape[0], GTMask.shape[1], 1))
    (h, w, l) = GTMask.shape
    bin_mask = GTMask != 0
    trimmed_mask = bin_mask[trim_px:-trim_px, trim_px:-trim_px, :]
    
    layer_npixels = np.sum(bin_mask, axis=(0,1))
    npixels = np.array([np.sum(layer_npixels[[lab==ftype for lab in layer_labels]]) for ftype in failure_types])
    
    trimmed_layer_npixels = np.sum(trimmed_mask, axis=(0,1))
    trimmed_npixels = [np.sum(trimmed_layer_npixels[[lab==ftype for lab in layer_labels]]) for ftype in failure_types]
    
    
    # Find bounding boxes
    
    # Filter out any layers without any marked pixels
    bin_mask_filtered = bin_mask[:,:,layer_npixels>0]
    layer_labels_filtered = layer_labels[layer_npixels>0]
    lf = len(layer_labels_filtered)
    
    # Sum along each axis
    sum0 = bin_mask_filtered.sum(axis=0)
    sum1 = bin_mask_filtered.sum(axis=1)
    
    bb = [None]*lf
    
    
    for i in range(lf):
        hor = np.where(sum0[:,i]>0)[0]
        ver = np.where(sum1[:,i]>0)[0]
        box_l = hor[0] if len(hor)>0 else 0
        box_r = hor[-1] if len(hor)>0 else 0
        box_t = ver[0] if len(ver)>0 else 0
        box_b = ver[-1] if len(ver)>0 else 0
        lab = failure_types.index(layer_labels_filtered[i])
        bb[i] = (box_l, box_r, box_t, box_b, lab)
        
    
    return (npixels, trimmed_npixels, tuple(bb))

def ReadImageData(file):
    
    if not os.path.exists(file):
        print(f'Error: Image {file} does not exist')
        return (0,)*9
    
    im = Image.open(file)
    im = np.array(im)
    im = im/255
    quantiles = np.quantile(im, [0.25, 0.5, 0.75])
    (h,w) = im.shape
    
    return (w,
            h, 
            np.mean(im), 
            np.std(im), 
            skew(im, axis=None),
            kurtosis(im, axis=None),
            quantiles[0], 
            quantiles[1], 
            quantiles[2],
            )
#%%

series = ['1', '2', '3', '4', '5', '6']


all_files = []
all_series = []
all_names = []
all_panel_ids = []
all_rows = []
all_cols = []
all_npixels = np.empty((0,4), dtype=int)
all_trimmed_npixels = np.empty((0,4), dtype=int)
all_bbs = []
all_image_data = np.empty((0,9))
all_image_GS_data = np.empty((0,9))

for ser in series:
    r = 3 if (ser=='1' or ser=='5') else 4
    c = 5 if (ser=='1' or ser=='5') else 6
    
    img_dir = f'Series{ser}/CellsCorr'
    img_GS_dir = f'Series{ser}/CellsGS'
    mask_dir = f'Series{ser}/MaskGT'
    
    files = [file for file in os.listdir(img_dir) if file.endswith('.png')]
    series = [ser]*len(files)
    names = [file[19:-4] for file in files]
    split_names = [name.split('_') for name in names]
    panel_ids = [spl[0]+'_'+spl[1] for spl in split_names]
    rows = [int(spl[r][3:]) for spl in split_names]
    cols = [int(spl[c]) for spl in split_names]
    mask_data = [ReadMaskData(os.path.join(mask_dir,f'GT_Serie_{ser}_Image_-{name}.mat')) for name in names]
    npixels = np.array([md[0] for md in mask_data])
    trimmed_npixels = np.array([md[1] for md in mask_data])
    bbs = [md[2] for md in mask_data]
    image_data = np.array([ReadImageData(os.path.join(img_dir, file)) for file in files])
    image_GS_data = np.array([ReadImageData(os.path.join(img_GS_dir, f'Serie_{ser}_ImageGS_-{name}.png')) for name in names])
    
    
    all_files.extend(files)
    all_series.extend(series)
    all_names.extend(names)
    all_panel_ids.extend(panel_ids)
    all_rows.extend(rows)
    all_cols.extend(cols)
    all_npixels = np.append(all_npixels, npixels, axis=0)
    all_trimmed_npixels = np.append(all_trimmed_npixels, trimmed_npixels, axis=0)
    all_bbs.extend(bbs)
    all_image_data = np.append(all_image_data, image_data, axis=0)
    all_image_GS_data = np.append(all_image_GS_data, image_GS_data, axis=0)

#%%
    # dictionary of lists  
dict = {'name': all_names,
        'series': all_series,
        'panel_id': all_panel_ids, 
        'row': all_rows, 
        'col': all_cols, 
        'npixels_crack_a': all_npixels[:,0],
        'npixels_crack_b': all_npixels[:,1],
        'npixels_crack_c': all_npixels[:,2],
        'npixels_finger_failure': all_npixels[:,3],
        'trimmed_npixels_crack_a': all_trimmed_npixels[:,0],
        'trimmed_npixels_crack_b': all_trimmed_npixels[:,1],
        'trimmed_npixels_crack_c': all_trimmed_npixels[:,2],
        'trimmed_npixels_finger_failure': all_trimmed_npixels[:,3],
        'bb': all_bbs,
        'width': all_image_data[:,0].astype(int),
        'height': all_image_data[:,1].astype(int),
        'mean': all_image_data[:,2],
        'std': all_image_data[:,3],
        'skew': all_image_data[:,4],
        'kurt': all_image_data[:,5],
        'q1': all_image_data[:,6],
        'median': all_image_data[:,7],
        'q3': all_image_data[:,8],
        'GS_mean': all_image_GS_data[:,2],
        'GS_std': all_image_GS_data[:,3],
        'GS_skew': all_image_GS_data[:,4],
        'GS_kurt': all_image_GS_data[:,5],
        'GS_q1': all_image_GS_data[:,6],
        'GS_median': all_image_GS_data[:,7],
        'GS_q3': all_image_GS_data[:,8],
        }  
       

df = pd.DataFrame(dict) 

#%%
# saving the dataframe 
df.to_csv('EL_data.csv', index=False)
    
#%%
df = pd.read_csv('EL_data.csv')

#%%

df['trimmed_coverage_crack_a'] = df['trimmed_npixels_crack_a']/((df['width']-2*trim_px)*(df['height']-2*trim_px))
df['trimmed_coverage_crack_b'] = df['trimmed_npixels_crack_b']/((df['width']-2*trim_px)*(df['height']-2*trim_px))
df['trimmed_coverage_crack_c'] = df['trimmed_npixels_crack_c']/((df['width']-2*trim_px)*(df['height']-2*trim_px))
df['trimmed_coverage_finger_failure'] = df['trimmed_npixels_finger_failure']/((df['width']-2*trim_px)*(df['height']-2*trim_px))


df['label_crack_a'] = (df['trimmed_coverage_crack_a'] >= 0.0010).astype(int)
df['label_crack_b'] = (df['trimmed_coverage_crack_b'] >= 0.0020).astype(int)
df['label_crack_c'] = (df['trimmed_coverage_crack_c'] >= 0.0020).astype(int)
df['label_finger_failure'] = (df['trimmed_coverage_finger_failure'] >= 0.0015).astype(int)

def getLabel(x):
    return 3 if x[2]==1 else (2 if x[1]==1 else (4 if x[3]==1 else (1 if x[0]==1 else 0)))

df['label'] = np.apply_along_axis(getLabel, 1, df[['label_crack_a','label_crack_b','label_crack_c','label_finger_failure']])

#%%
#remove rows /filtering
bad = np.zeros(len(df), dtype=bool)

bad[df['height'] < 250] = True
bad[df['width'] < 250] = True
bad[df['GS_mean'] < 0.30] = True

# These images are cropped incorrectly
bad[(df['series']==2) & (df['panel_id']=='7_4023') & (df['col']==6)] = True
bad[(df['series']==4) & (df['panel_id']=='2_4169') & (df['col']==1)] = True
bad[(df['series']==4) & (df['panel_id']=='12_4164') & (df['col']==1)] = True
bad[(df['series']==6) & (df['panel_id']=='1_4258') & (df['col']==1)] = True
bad[(df['series']==6) & (df['panel_id']=='11_4253') & (df['col']==1)] = True
bad[(df['series']==6) & (df['panel_id']=='12_4272') & (df['col']==1)] = True
bad[(df['series']==4) & (df['panel_id']=='23_4110') & (df['col']>=5)] = True
bad[(df['series']==6) & (df['panel_id']=='18_4249') & (df['col']>=4)] = True

df = df[bad==False]
df.reset_index(drop=True, inplace=True)

#%%

test_size = 0.20
val_size = 0.20/(1-0.20)

N = len(df)
random.seed(0)
np.random.seed(0)
train_val_ids, test_ids = train_test_split(np.arange(N), test_size=test_size, random_state=0, stratify=df['label'])
train_ids, val_ids = train_test_split(train_val_ids, test_size=val_size, random_state=1, stratify=df['label'][train_val_ids])

split = np.zeros(N, dtype=np.int8)
split[train_ids] = 0
split[val_ids] = 1
split[test_ids] = 2
df['split'] = split

df['kfold_outer'] = -1

kf = KFold(5, shuffle=True, random_state=2)
for fold, (_, ids) in enumerate(kf.split(train_val_ids)):
    df.loc[train_val_ids[ids], 'kfold_outer'] = fold
    
df['kfold_inner'] = -1

kf2 = KFold(5, shuffle=True, random_state=3)
for fold, (_, ids) in enumerate(kf2.split(train_ids)):
    df.loc[train_ids[ids], 'kfold_inner'] = fold


#%%
df.to_csv('EL_data_split.csv', index=False)
