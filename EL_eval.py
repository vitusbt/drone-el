
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sklearn.metrics as M
from scipy.special import softmax

#%%
def EvalModelMetrics(model_output):
    train_loss = model_output['train_loss']
    test_loss = model_output['test_loss']
    test_outputs = softmax(model_output['test_outputs'], axis=2)
    train_labels = model_output['train_labels'].astype(int)
    test_labels = model_output['test_labels'].astype(int)
    
    if len(train_labels.shape) == 3:
        train_labels = np.argmax(train_labels, axis=-1)
    if len(test_labels.shape) == 3:
        test_labels = np.argmax(test_labels, axis=-1)

    test_bin_labels = (test_labels > 0).astype(int)
    test_pred = np.argmax(test_outputs, axis=-1)
    test_bin_pred = (test_pred > 0).astype(int)
    
    n_epochs, n_samples, n_classes = test_outputs.shape

    
    test_log_outputs_all = np.log(test_outputs)
    test_log_outputs_true = np.zeros((n_epochs, n_samples))
    weights = np.zeros((n_epochs, n_samples))
    for i in range(n_epochs):
        test_log_outputs_true[i] = test_log_outputs_all[i,np.arange(n_samples),test_labels[i]]
        weights[i] = 1/(n_classes*(np.array([np.sum(test_labels[i]==c) for c in range(n_classes)]))[test_labels[i]])
    
    np.array([np.sum(test_labels==i, axis=1) for i in range(5)]).transpose()
    
    confusion = np.zeros((n_epochs, n_classes, n_classes), dtype=int)
    
    prec = np.zeros((n_epochs, n_classes))
    recall = np.zeros((n_epochs, n_classes))
    f1 = np.zeros((n_epochs, n_classes))
    bal_acc = np.zeros(n_epochs)
    accuracy = np.zeros(n_epochs)
    
    prec_micro = np.zeros(n_epochs)
    recall_micro = np.zeros(n_epochs)
    f1_micro = np.zeros(n_epochs)
    
    prec_macro = np.zeros(n_epochs)
    recall_macro = np.zeros(n_epochs)
    f1_macro = np.zeros(n_epochs)
    
    prec_weighted = np.zeros(n_epochs)
    recall_weighted = np.zeros(n_epochs)
    f1_weighted = np.zeros(n_epochs)
    
    roc_auc_ovo = np.zeros(n_epochs)
    roc_auc_ovr = np.zeros(n_epochs)
    kappa = np.zeros(n_epochs)
    mcc = np.zeros(n_epochs)
    
    bin_confusion = np.zeros((n_epochs, 2, 2))
    bin_prec = np.zeros(n_epochs)
    bin_recall = np.zeros(n_epochs)
    bin_f1 = np.zeros(n_epochs)
    
    for ep in range(n_epochs):
        confusion[ep] = M.confusion_matrix(test_labels[ep], test_pred[ep])
        prec[ep], recall[ep], f1[ep], _ = M.precision_recall_fscore_support(test_labels[ep], test_pred[ep])
        prec_micro[ep], recall_micro[ep], f1_micro[ep], _ = M.precision_recall_fscore_support(test_labels[ep], test_pred[ep], labels = [1,2,3,4], average='micro')
        prec_macro[ep], recall_macro[ep], f1_macro[ep], _ = M.precision_recall_fscore_support(test_labels[ep], test_pred[ep], labels = [1,2,3,4], average='macro')
        prec_weighted[ep], recall_weighted[ep], f1_weighted[ep], _ = M.precision_recall_fscore_support(test_labels[ep], test_pred[ep], labels = [1,2,3,4], average='weighted')
        
        roc_auc_ovo[ep] = M.roc_auc_score(test_labels[ep], test_outputs[ep], multi_class='ovo')
        roc_auc_ovr[ep] = M.roc_auc_score(test_labels[ep], test_outputs[ep], multi_class='ovr')
        kappa[ep] = M.cohen_kappa_score(test_labels[ep], test_pred[ep])
        mcc[ep] = M.matthews_corrcoef(test_labels[ep], test_pred[ep])
        
        bal_acc[ep] = M.balanced_accuracy_score(test_labels[ep], test_pred[ep], adjusted=True)
        accuracy[ep] = M.accuracy_score(test_labels[ep], test_pred[ep])
        bin_confusion = M.confusion_matrix(test_bin_labels[ep], test_bin_pred[ep])
        bin_prec[ep], bin_recall[ep], bin_f1[ep], _ = M.precision_recall_fscore_support(test_bin_labels[ep], test_bin_pred[ep], average='binary')
        
    return {'train_loss': train_loss,
            'test_loss': test_loss,
            'confusion': confusion, 
            'precision': prec,
            'recall': recall, 
            'f1': f1,
            'precision_micro': prec_micro,
            'recall_micro': recall_micro, 
            'f1_micro': f1_micro,
            'precision_macro': prec_macro,
            'recall_macro': recall_macro, 
            'f1_macro': f1_macro,
            'precision_weighted': prec_weighted,
            'recall_weighted': recall_weighted, 
            'f1_weighted': f1_weighted,
            'roc_auc_ovo': roc_auc_ovo,
            'roc_auc_ovr': roc_auc_ovr,
            'kappa': kappa,
            'mcc': mcc,
            'accuracy': accuracy,
            'bal_acc': bal_acc,
            'bin_confusion': bin_confusion,
            'bin_precision': bin_prec,
            'bin_recall': bin_recall,
            'bin_f1': bin_f1,
            }

#%%
if __name__ == '__main__':
    
    # For plots
    dpi = 600
    save = True
    save_path = 'plots/'
    save_filetype = 'png'

    save_prefix = 'model_transfer'

    path = 'EL_model_results/'
    
    model_files = ['model_sq10_outputs.npz',
                    'model_sq11_outputs.npz',
                    'model_vgg11_outputs.npz',
                    'model_vgg13_outputs.npz',
                    'model_resnet18_outputs.npz',
                    'model_resnet34_outputs.npz',
                    ]


    model_names = ['SqueezeNet 1.0',
                    'SqueezeNet 1.1',
                    'VGG-11',
                    'VGG-13',
                    'ResNet-18',
                    'ResNet-34',
                    ]

    class_names = ['No defect', 'Crack A', 'Crack B', 'Crack C', 'Finger failure']

    N = len(model_names)
    model_outputs = [np.load(path+model_files[i]) for i in range(N)]
    

    met = [EvalModelMetrics(model_outputs[i]) for i in range(N)]
    
    baseline_acc = np.mean(model_outputs[0]['test_labels'][0]==0)
    
    #%%
    class_cols = ['#404040', '#00c000', '#ffc000', '#ff4040', '#0040ff']
    mpl.rcParams["axes.prop_cycle"] = cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#000000', '#ff0000', '#00ff00', '#0000ff'])
    
    def smooth(x, alpha=0.25, logscale=False):
        if not logscale:
            return np.array(pd.DataFrame(x).ewm(alpha=alpha).mean()).flatten()
        else:
            return np.array(np.exp(pd.DataFrame(np.log(x)).ewm(alpha=alpha).mean())).flatten()
        
    def plot_metrics(metric, title, met_list=met, model_names=model_names, smoothing_alpha=0.25, ylim_low=None, ylim_high=None, log=False, hline=None, linewidth=1, figsize=(8,6), dpi=dpi, save=save, save_path=save_path, save_prefix=save_prefix):
        plt.figure(figsize=figsize, dpi=dpi)
        plt.grid()
        plt.title(title)
        for i in range(len(met_list)):
            plt.plot(np.arange(1,len(met_list[i][metric])+1), smooth(met_list[i][metric], smoothing_alpha, logscale=log), label=model_names[i], linewidth=linewidth)
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
        if save:
            plt.savefig(save_path + save_prefix + '_' + metric + '.' + save_filetype, bbox_inches='tight')
        plt.show()
        
    def plot_class_metrics(metric, title, met_list=met, model_names=model_names, class_names=class_names, class_cols=class_cols, smoothing_alpha=0.25, ylim_low=None, ylim_high=None, log=False, hline=None, linewidth=1, figsize=(8,6), dpi=dpi, save=save, save_path=save_path, save_prefix=save_prefix):
        for i in range(len(met_list)):
            plt.figure(figsize=figsize, dpi=dpi)
            plt.grid()
            plt.title(title + ", " + model_names[i])
            for j in range(len(class_names)):
                plt.plot(np.arange(1,len(met_list[i][metric][:,j])+1), met_list[i][metric][:,j], label=class_names[j], linewidth=linewidth, color=class_cols[j])
            if log:
                plt.yscale('log')
            if ylim_low:
                plt.ylim(bottom=ylim_low)
            if ylim_high:
                plt.ylim(top=ylim_high)
            if hline:
                plt.axhline(y=hline, color='black', linestyle='--', linewidth=1.5, label='Baseline')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            if save:
                plt.savefig(save_path + save_prefix + '_' + metric + '_model' + str(i) + '.' + save_filetype, bbox_inches='tight')
            plt.show()
            
    def plot_matrix(mat):
        fig, ax = plt.subplots()
        ax.matshow(mat)
        for i in range(5):
            for j in range(5):
                c = mat[j,i]
                ax.text(i, j, f'{c}', va='center', ha='center')
    
    
    plot_metrics('accuracy', 'Accuracy', hline=baseline_acc, ylim_low=0.959, ylim_high=0.981, met_list=met, model_names=model_names)
    plot_metrics('f1_macro', 'Macro F1', met_list=met, model_names=model_names)
    plot_metrics('precision_macro', 'Macro precision', met_list=met, model_names=model_names)
    plot_metrics('recall_macro', 'Macro recall', met_list=met, model_names=model_names)
    #plot_metrics('bin_f1', 'Binary F1')
    #plot_metrics('bin_precision', 'Binary precision')
    #plot_metrics('bin_recall', 'Binary recall')
    plot_metrics('mcc', 'Matthew\'s correlation coefficient', met_list=met, model_names=model_names)
    plot_metrics('train_loss', 'Training loss', log=True, met_list=met, model_names=model_names)
    plot_metrics('test_loss', 'Validation loss', log=True, met_list=met, model_names=model_names)
    
    plot_class_metrics('f1', 'Per-class F1', figsize=(16/3, 4))
    
    
    #%%
    
    def exp_weighted_mean(x, com=50):
        alpha = com/(com+1)
        w = np.flip(alpha**np.arange(len(x)))
        w /= w.sum()
        return np.sum(x*w)
    
    metrics = ['f1_macro','precision_macro','recall_macro','accuracy','mcc']
    
    metrics_dict = {'model': model_names}
    metrics_dict.update({metric+'_50': np.array([exp_weighted_mean(m[metric], com=50) for m in met]) for metric in metrics})
    metrics_table = pd.DataFrame(metrics_dict)
    
    print(metrics_table.to_string(index=False,float_format='%.4f'))
    
    
        