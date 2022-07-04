
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import random

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from scipy.special import softmax

import sklearn.metrics as M

from EL_eval import EvalModelMetrics
from EL_eval_cp_inner import E, GetPredSets
from EL_data_visualization import LoadImage


if __name__ == '__main__':
    
    # Load data
    all_data = pd.read_csv('EL_data_split.csv')
    
    seeds = 50
    epoch = 125
    alpha = 0.03

    path = 'EL_cp_outer_results/'
    model_files = ['cp_outer_k0_outputs.npz',
                    'cp_outer_k1_outputs.npz',
                    'cp_outer_k2_outputs.npz',
                    'cp_outer_k3_outputs.npz',
                    'cp_outer_k4_outputs.npz',]

    base_model_file = 'cp_outer_all_outputs.npz'

    K = 5
    kfold_data = [all_data[all_data['kfold_outer']==i] for i in range(K)]
    test_data = all_data[all_data['split']==2]

    model_outputs = [np.load(path+model_files[i]) for i in range(K)]
    base_model_outputs = np.load(path + base_model_file)

    for i in range(5):
        kfold_data[i].reset_index(drop=True,inplace=True)
    test_data.reset_index(drop=True,inplace=True)
    
    N_calib = np.array([len(kfold_data[i]) for i in range(K)])
    N_train = N_calib.sum()
    N_test = len(test_data)

    calib_labels = np.concatenate([model_outputs[i]['test_labels'][-1,N_test:] for i in range(5)], axis=0).astype(int)
    calib_outputs = np.concatenate([softmax(model_outputs[i]['test_outputs'][epoch-1,N_test:], axis=-1) for i in range(5)], axis=-2)

    test_labels = model_outputs[0]['test_labels'][-1,:N_test].astype(int)
    test_outputs = np.array([softmax(model_outputs[i]['test_outputs'][epoch-1,:N_test], axis=-1) for i in range(5)])
    test_outputs_ensemble = test_outputs.mean(axis=0)
    test_outputs_base = softmax(model_outputs[i]['test_outputs'][epoch-1,:N_test], axis=-1)
    
    test_pred = np.argmax(test_outputs, axis=-1)
    test_pred_ensemble = np.argmax(test_outputs_ensemble, axis=-1)
    test_pred_base = np.argmax(test_outputs_base, axis=-1)
    
    base_metrics = EvalModelMetrics(base_model_outputs)
        
    #%%
    
    prediction_sets = np.zeros((seeds, N_test, 5), dtype=bool)
    
    for seed in range(seeds):
    
        random.seed(seed)
        np.random.seed(seed)
        
        #U_calib = np.full(N_train, 0.5)
        #U_test = np.full(N_test, 0.5)
        U_calib = np.random.rand(N_train)
        U_test = np.random.rand(N_test)

        
        E_ = E(U_calib, calib_outputs)
        E_ = E_[np.arange(N_train), calib_labels.astype(int)]
        
        E_test = np.zeros((5, N_test, 5)) # nfolds x ntest x nclasses
        for k in range(5):
            E_test[k] = E(U_test, test_outputs[k])
        
        prediction_sets[seed] = GetPredSets(E_test, E_, N_calib, alpha)

    #%%
    
    # Division where dividing by 0 returns 0
    def div(x1,x2):
        return np.divide(x1,x2, out=np.zeros_like(x1, dtype=float), where=x2!=0)
    
    set_sizes = prediction_sets.sum(axis=-1)
    confident = set_sizes == 1
    covered = prediction_sets[:,np.arange(N_test),test_labels]
    
    tp_raw = np.stack([((test_labels==c)[np.newaxis,:] & prediction_sets[:,:,c]) for c in range(5)], axis=-1)
    fp_raw = np.stack([(~(test_labels==c)[np.newaxis,:] & prediction_sets[:,:,c]) for c in range(5)], axis=-1)
    fn_raw = np.stack([((test_labels==c)[np.newaxis,:] & ~prediction_sets[:,:,c]) for c in range(5)], axis=-1)
    tn_raw = np.stack([(~(test_labels==c)[np.newaxis,:] & ~prediction_sets[:,:,c]) for c in range(5)], axis=-1)
    
    tp = tp_raw.sum(axis=-2)
    fp = fp_raw.sum(axis=-2)
    fn = fn_raw.sum(axis=-2)
    tn = tn_raw.sum(axis=-2)
    
    tp_confident = (tp_raw & confident[:,:,np.newaxis]).sum(axis=-2)
    fp_confident = (fp_raw & confident[:,:,np.newaxis]).sum(axis=-2)
    fn_confident = (fn_raw & confident[:,:,np.newaxis]).sum(axis=-2)
    tn_confident = (tn_raw & confident[:,:,np.newaxis]).sum(axis=-2)
    
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
    
    macro_precision = precision[1:].mean(axis=-1)
    macro_recall = recall[1:].mean(axis=-1)
    macro_f1 = f1[1:].mean(axis=-1)
    
    macro_precision_confident = precision_confident[1:].mean(axis=-1)
    macro_recall_confident = recall_confident[1:].mean(axis=-1)
    macro_f1_confident = f1_confident[1:].mean(axis=-1)
    
    macro_precision_confident_allseeds = precision_confident_allseeds[:,1:].mean(axis=-1)
    macro_recall_confident_allseeds = recall_confident_allseeds[:,1:].mean(axis=-1)
    macro_f1_confident_allseeds = f1_confident_allseeds[:,1:].mean(axis=-1)
    
    predictions_confident = prediction_sets & confident[:,:,np.newaxis]
    
    confusion_matrix_confident = np.zeros((5,5,seeds),dtype=int)
    for i in range(5):
        for j in range(5):
            confusion_matrix_confident[i,j] = np.sum(predictions_confident[:,:,j] & (test_labels == i), axis=-1)
    
#%%

    def smooth(x, alpha=0.25, logscale=False):
        if not logscale:
            return np.array(pd.DataFrame(x).ewm(alpha=alpha).mean()).flatten()
        else:
            return np.array(np.exp(pd.DataFrame(np.log(x)).ewm(alpha=alpha).mean())).flatten()
    

    def plot_metric(x, title, smoothing_alpha=0.25, ylim_low=None, ylim_high=None, log=False, hline=None, figsize=(8,6), dpi=300):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.grid()
        plt.title(title)
        plt.plot(np.arange(1,len(x)+1), x, label='Raw', color='tab:blue', alpha=0.5, linewidth=1)
        plt.plot(np.arange(1,len(x)+1), smooth(x, smoothing_alpha, logscale=log), color='tab:blue', label='Smoothed', linewidth=2)
        if log:
            plt.yscale('log')
        if ylim_low:
            plt.ylim(bottom=ylim_low)
        if ylim_high:
            plt.ylim(top=ylim_high)
        if hline:
            plt.axhline(y=hline, color='black', linestyle='--', linewidth=1.5, label='Baseline')
        plt.xlabel('Epoch')
        plt.legend()
        plt.show()

    base_f1 = base_metrics['f1_macro'][:300]
    base_precision= base_metrics['precision_macro'][:300]
    base_recall = base_metrics['recall_macro'][:300]
    base_accuracy = base_metrics['accuracy'][:300]
    baseline_acc = np.mean(test_labels==0)
    
    plot_metric(base_f1, 'Macro $F_1$ score on test set')
    plot_metric(base_precision, 'Macro precision on test set')
    plot_metric(base_recall, 'Macro recall on test set')
    plot_metric(base_accuracy, 'Accuracy on test set', ylim_low=0.959, ylim_high=0.981, hline=baseline_acc)
    
    # Metrics of standard classifier after 300 epochs
    ep = 299
    print('Simple: ')
    print(base_metrics['confusion'][ep])
    print('F1 = ' + str(base_metrics['f1'][ep]))
    print('Precision = ' + str(base_metrics['precision'][ep]))
    print('Recall = ' + str(base_metrics['recall'][ep]))
    print('Macro F1 = ' + str(base_metrics['f1_macro'][ep]))
    print('Macro precision = ' + str(base_metrics['precision_macro'][ep]))
    print('Macro recall = ' + str(base_metrics['recall_macro'][ep]))
    print('Accuracy = ' + str(base_metrics['accuracy'][ep]))
    print('MCC = ' + str(base_metrics['mcc'][ep]))
    
    print()
    
    # Metrics of (median) conformal classifier
    median_id = np.argsort(macro_f1_confident_allseeds)[(seeds-1)//2]
    confident_true = test_labels[confident[median_id]]
    confident_pred = np.argmax(prediction_sets[median_id,confident[median_id]],axis=-1)
    print('Conformal: ')
    print(confusion_matrix_confident[:,:,median_id])
    print('F1 = ' + str(f1_confident_allseeds[median_id]))
    print('Precision = ' + str(precision_confident_allseeds[median_id]))
    print('Recall = ' + str(recall_confident_allseeds[median_id]))
    print('Macro F1 = ' + str(macro_f1_confident_allseeds[median_id]))
    print('Macro precision = ' + str(macro_precision_confident_allseeds[median_id]))
    print('Macro recall = ' + str(macro_recall_confident_allseeds[median_id]))
    acc = np.diag(confusion_matrix_confident[:,:,median_id]).sum()/np.sum(confusion_matrix_confident[:,:,median_id])
    print('Accuracy = ' + str(acc))
    print('MCC = ' + str(M.matthews_corrcoef(confident_true, confident_pred)))
    
#%% Visualization

    class_names = np.array(['No defect', 'Crack A', 'Crack B', 'Crack C', 'Finger failure'])

    def PredSetsToList(sets):
        return [np.arange(5)[p].tolist() for p in sets]
    
    def SetToString(ps):
        string = '{'
        for (i, c) in enumerate(ps):
            string += class_names[c]
            if i < len(ps) - 1:
                string += ', '
                if len(ps)>=3 and i==len(ps)-3:
                    string+='\n'
        string += '}'
        return string
    
    def ShowImages(dataset, pred_sets, n=10):
        ids = np.random.choice(len(dataset), size=n, replace=False)
        print(ids)
        imgs = [LoadImage(dataset, ids[i]) for i in range(n)]
        labels = dataset['label'][ids].tolist()
        ps = PredSetsToList(pred_sets[ids])
        for i in range(n):
            plt.figure(dpi=300)
            plt.axis('off')
            plt.title('Label = ' + str(class_names[labels[i]]) + '\nPS = ' + SetToString(ps[i]))
            plt.imshow(imgs[i])
            plt.show()
            
    confident_correct_allseeds = confident & covered
    confident_incorrect_allseeds = confident & ~covered
    inconfident_correct_allseeds = ~confident & covered
    inconfident_incorrect_allseeds = ~confident & ~covered
    
    confident_correct = confident_correct_allseeds[median_id]
    confident_incorrect = confident_incorrect_allseeds[median_id]
    inconfident_correct = inconfident_correct_allseeds[median_id]
    inconfident_incorrect = inconfident_incorrect_allseeds[median_id]
    
    prediction_sets_confident_correct = prediction_sets[median_id,confident_correct]
    prediction_sets_confident_incorrect = prediction_sets[median_id,confident_incorrect]
    prediction_sets_inconfident_correct = prediction_sets[median_id,inconfident_correct]
    prediction_sets_inconfident_incorrect = prediction_sets[median_id,inconfident_incorrect]
    
    
    test_data_confident_correct = test_data[confident_correct]
    test_data_confident_incorrect = test_data[confident_incorrect]
    test_data_inconfident_correct = test_data[inconfident_correct]
    test_data_inconfident_incorrect = test_data[inconfident_incorrect]
    
    test_data_confident_correct.reset_index(drop=True,inplace=True)
    test_data_confident_incorrect.reset_index(drop=True,inplace=True)
    test_data_inconfident_correct.reset_index(drop=True,inplace=True)
    test_data_inconfident_incorrect.reset_index(drop=True,inplace=True)

    # Visualization
    plt.gray()
    np.random.seed(2022)
    
    n=20
    ShowImages(test_data_confident_correct, prediction_sets_confident_correct, n)
    ShowImages(test_data_confident_incorrect, prediction_sets_confident_incorrect, n)
    ShowImages(test_data_inconfident_correct, prediction_sets_inconfident_correct, n)
    ShowImages(test_data_inconfident_incorrect, prediction_sets_inconfident_incorrect, n)
    
    
    