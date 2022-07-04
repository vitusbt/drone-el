
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import pandas as pd

import random

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from scipy.special import softmax
from scipy.ndimage import gaussian_filter1d

from EL_eval import EvalModelMetrics

def E(u, pi):
    # calculate for all y at the same time
    n = pi.shape[0]
    argsortpi = np.flip(np.argsort(pi, axis=1), axis=1)
    sortedpi = pi[np.arange(n)[:,np.newaxis] ,argsortpi]
    cumsumpi = np.cumsum(sortedpi, axis=1)
    indices = np.full(pi.shape, np.arange(pi.shape[1]))
    indices[np.arange(n)[:,np.newaxis], argsortpi] = indices.copy()
    tau = cumsumpi[np.arange(n)[:,np.newaxis], indices]
    return tau - u[:,np.newaxis]*pi

def GetPredSets(E_test, E_calib, N_calib, alphas):
    
    # E_test_samples:       (nfolds x ntest x nclasses)
    # E_to_compare[fold]:   (nclasses)
    # E_:                   (ntrain)
    # amounts:              (ntest x nclasses)
    # argsort_amounts:      (ntest x nclasses)
    # ps_size:              (ntest)
    # ps:                   (ntest x nclasses)
    
    K, N_test, C = E_test.shape
    N_train = len(E_calib)
    amounts = np.zeros((N_test, C), dtype=int)
    counter = 0
    for c in range(C):
        amounts += (E_calib[counter:(counter + N_calib[c]), np.newaxis, np.newaxis] < E_test[np.newaxis, c, :, :]).sum(axis=0)
        counter += N_calib[c]
    threshold = (1-alphas)*(N_train+1)
    if np.ndim(alphas) == 0:
        return (amounts < threshold)
    return (amounts[np.newaxis,:,:] < threshold[:,np.newaxis,np.newaxis])





if __name__ == '__main__':
    
    # Load data
    all_labels = pd.read_csv('EL_data_split.csv')

    seeds = 50
    epochs = np.arange(25, 501, 25)
    alphas = np.linspace(0.005, 0.020, 40)
    N_alphas = len(alphas)
    N_epochs = len(epochs)

    path = 'EL_cp_inner_results/'
    model_files = ['cp_inner_k0_outputs.npz',
                    'cp_inner_k1_outputs.npz',
                    'cp_inner_k2_outputs.npz',
                    'cp_inner_k3_outputs.npz',
                    'cp_inner_k4_outputs.npz',]

    base_model_file = 'cp_inner_all_outputs.npz'

    K = 5
    kfold_labels = [all_labels[all_labels['kfold_inner']==i] for i in range(K)]
    test_labels = all_labels[all_labels['split']==1]

    model_outputs = [np.load(path+model_files[i]) for i in range(K)]
    base_model_outputs = np.load(path + base_model_file)

    for i in range(5):
        kfold_labels[i].reset_index(drop=True,inplace=True)
    test_labels.reset_index(drop=True,inplace=True)
    
    N_calib = np.array([len(kfold_labels[i]) for i in range(K)])
    N_train = N_calib.sum()
    N_test = len(test_labels)

    calib_labels = np.concatenate([model_outputs[i]['test_labels'][-1,N_test:] for i in range(5)], axis=0).astype(int)
    calib_outputs = np.concatenate([softmax(model_outputs[i]['test_outputs'][epochs-1,N_test:], axis=-1) for i in range(5)], axis=1)

    test_labels = model_outputs[0]['test_labels'][-1,:N_test].astype(int)
    test_outputs = np.array([softmax(model_outputs[i]['test_outputs'][epochs-1,:N_test], axis=-1) for i in range(5)])
    test_pred = np.argmax(test_outputs, axis=-1)

    base_metrics = EvalModelMetrics(base_model_outputs)
    
    
        
    #%%
    
    # Huge array containing all prediction sets across all seeds, epochs and alphas
    prediction_sets = np.zeros((seeds, N_epochs, N_alphas, N_test, 5), dtype=bool)
    
    for seed in range(seeds):
    
        random.seed(seed)
        np.random.seed(seed)
        
        #U_calib = np.full(N_train, 0.5)
        #U_test = np.full(N_test, 0.5)
        U_calib = np.random.rand(N_train)
        U_test = np.random.rand(N_test)

        for ep in range(N_epochs):
            E_ = E(U_calib, calib_outputs[ep])
            E_ = E_[np.arange(N_train), calib_labels.astype(int)]
    
            
            E_test = np.zeros((5, N_test, 5)) # nfolds x ntest x nclasses
            for k in range(5):
                E_test[k] = E(U_test, test_outputs[k,ep])
            
            prediction_sets[seed,ep] = GetPredSets(E_test, E_, N_calib, alphas)


    #%%
    
    # Division where dividing by 0 returns 0
    def div(x1,x2):
        return np.divide(x1,x2, out=np.zeros_like(x1, dtype=float), where=x2!=0)
    
    set_sizes = prediction_sets.sum(axis=-1)
    confident = set_sizes == 1
    covered = prediction_sets[:,:,:,np.arange(N_test),test_labels]
    
    tp_raw = np.stack([((test_labels==c)[np.newaxis,np.newaxis,np.newaxis,:] & prediction_sets[:,:,:,:,c]) for c in range(5)], axis=-1)
    fp_raw = np.stack([(~(test_labels==c)[np.newaxis,np.newaxis,np.newaxis,:] & prediction_sets[:,:,:,:,c]) for c in range(5)], axis=-1)
    fn_raw = np.stack([((test_labels==c)[np.newaxis,np.newaxis,np.newaxis,:] & ~prediction_sets[:,:,:,:,c]) for c in range(5)], axis=-1)
    tn_raw = np.stack([(~(test_labels==c)[np.newaxis,np.newaxis,np.newaxis,:] & ~prediction_sets[:,:,:,:,c]) for c in range(5)], axis=-1)
    
    tp = tp_raw.sum(axis=-2)
    fp = fp_raw.sum(axis=-2)
    fn = fn_raw.sum(axis=-2)
    tn = tn_raw.sum(axis=-2)
    
    tp_confident = (tp_raw & confident[:,:,:,:,np.newaxis]).sum(axis=-2)
    fp_confident = (fp_raw & confident[:,:,:,:,np.newaxis]).sum(axis=-2)
    fn_confident = (fn_raw & confident[:,:,:,:,np.newaxis]).sum(axis=-2)
    tn_confident = (tn_raw & confident[:,:,:,:,np.newaxis]).sum(axis=-2)
    
    precision_allseeds = div(tp, tp+fp)
    recall_allseeds = div(tp, tp+fn)
    f1_allseeds = 2 * div(precision_allseeds * recall_allseeds, precision_allseeds + recall_allseeds)
    
    precision_confident_allseeds = div(tp_confident, tp_confident+fp_confident)
    recall_confident_allseeds = div(tp_confident, tp_confident+fn_confident)
    f1_confident_allseeds = 2 * div(precision_confident_allseeds * recall_confident_allseeds, precision_confident_allseeds + recall_confident_allseeds)
    
    confident_proportion_allseeds = confident.mean(axis=-1)
    confident_proportion = confident_proportion_allseeds.mean(axis=0)
    
    precision = precision_allseeds.mean(axis=0)
    recall = recall_allseeds.mean(axis=0)
    f1 = f1_allseeds.mean(axis=0)
    
    precision_confident = precision_confident_allseeds.mean(axis=0)
    recall_confident = recall_confident_allseeds.mean(axis=0)
    f1_confident = f1_confident_allseeds.mean(axis=0)
    
    macro_precision = precision[:,:,1:].mean(axis=-1)
    macro_recall = recall[:,:,1:].mean(axis=-1)
    macro_f1 = f1[:,:,1:].mean(axis=-1)
    
    macro_precision_confident = precision_confident[:,:,1:].mean(axis=-1)
    macro_recall_confident = recall_confident[:,:,1:].mean(axis=-1)
    macro_f1_confident = f1_confident[:,:,1:].mean(axis=-1)
    
    predictions_confident = prediction_sets & confident[:,:,:,:,np.newaxis]
    
    confusion_matrix_confident = np.zeros((5,5,seeds,N_epochs,N_alphas),dtype=int)
    for i in range(5):
        for j in range(5):
            confusion_matrix_confident[i,j] = np.sum(predictions_confident[:,:,:,:,j] & (test_labels == i), axis=-1)
    
    #%%
    
    X,Y = np.meshgrid(alphas, epochs)
    
    def plotgrid(mat, title=None, zero_center=False):
        plt.figure(dpi=600, figsize=(6,4))
        if zero_center:
            plt.set_cmap('seismic')
            norm = colors.TwoSlopeNorm(vmin=-np.max(np.abs(mat)), vcenter=0, vmax=np.max(np.abs(mat)))
            plt.pcolormesh(X, Y, mat, norm=norm)
        else:
            plt.set_cmap('viridis')
        plt.pcolormesh(X, Y, mat)
        plt.colorbar()
        plt.xlabel(r'$\alpha$')
        plt.ylabel('Epoch')
        if title:
            plt.title(title)
        plt.show()

    macro_f1_tilde = macro_f1_confident*confident_proportion
    
    macro_f1_confident_filtered = gaussian_filter1d(macro_f1_confident, sigma=1, axis=0, mode='nearest')
    plotgrid(macro_precision_confident, 'Macro precision on confident predictions')
    plotgrid(macro_recall_confident, 'Macro recall on confident predictions')
    plotgrid(macro_f1_confident, r'Macro $F_1$ on confident predictions')
    plotgrid(macro_f1_confident_filtered, r'Macro $F_1$ on confident predictions, filtered')
    
    macro_f1_tilde_filtered = macro_f1_confident_filtered * confident_proportion
    plotgrid(confident_proportion, 'Confidence proportion')
    plotgrid(macro_f1_tilde_filtered, r'Estimated $\tilde{F}_1$ value')
    
    np.unravel_index(np.argmax(macro_f1_tilde_filtered), macro_f1_tilde.shape)
    
    
    base_f1 = base_metrics['f1_macro']
    base_f1_filtered = gaussian_filter1d(base_metrics['f1_macro'], sigma=25, mode='nearest')
    plt.figure(dpi=600, figsize=(8,6))
    plt.grid()
    plt.title(r'Macro $F_1$ score')
    plt.plot(np.arange(1,len(base_f1)+1), base_f1, label=r'Raw', color='tab:blue', alpha=0.5, linewidth=1)
    plt.plot(np.arange(1,len(base_f1)+1), base_f1_filtered, label=r'Filtered', color='tab:blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    np.argmax(base_f1_filtered)
    np.max(base_f1_filtered)
    base_f1[np.argmax(base_f1_filtered)]
  