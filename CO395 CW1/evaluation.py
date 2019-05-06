import numpy as np
import plotTree
from decisionTree import DecisionTree
import matplotlib.pyplot as plt
# helper function
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    Plot the confusion matrix
    '''
    print(title)
    print(cm)
    #classes = np.unique(actual).astype(int)
    classes = [1,2,3,4]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '2'
    thresh = cm.max() / 2.

    for i, j in ((i,j) for i in range(cm.shape[0]) for j in range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    fold_size = dataset.shape[0] / n_folds
    idx = np.random.permutation(dataset.shape[0])
    fold_idx = np.split(idx, n_folds)
    return fold_idx


# Evaluate an algorithm using a cross validation split
def cross_valid(dataset, fold_idx):
    i = 0
    n_label = len(np.unique(dataset[:,-1]))
    cm = np.zeros((n_label,n_label))
    for valid_idx in fold_idx:
        i += 1
        valid_set = dataset[valid_idx,:]
        train_idx = np.delete(np.arange(dataset.shape[0]), valid_idx)
        train_set = dataset[train_idx,:]
        dtree = DecisionTree(train_set[:,:7],train_set[:,-1])
        cm += dtree.confusion_matrix(valid_set)
    cm /= len(fold_idx)
    return cm

def cross_valid_prune(dataset, fold_idx):
    i = 0
    n_label = len(np.unique(dataset[:,-1]))
    cm = np.zeros((n_label,n_label))
    for valid_idx in fold_idx:
        i += 1
        prune_index = fold_idx[(i)%10]
        prune_set = dataset[prune_index,:]
        valid_set = dataset[valid_idx,:]
        delete_index = np.concatenate([prune_index, valid_idx])
        train_idx = np.delete(np.arange(dataset.shape[0]), delete_index)
        train_set = dataset[train_idx,:]
        dtree = DecisionTree(train_set[:,:7],train_set[:,-1])
        dtree.post_prune(prune_set)

        cm += dtree.confusion_matrix(valid_set)
    cm /= len(fold_idx)
    return cm


def evaluation_metrics(cm):
    plot_confusion_matrix(cm)
    np.set_printoptions(precision=5)
    print("Recall: ",recalls(cm), "\n","Precision: ", precisions(cm), "\n",
        "F1 score: ", F1Measure(cm), "\n","Classification_rate: ", '%.5f' % classification_rate(cm))

def recalls(cm):
    size = len(cm)
    recalls = np.zeros(size)
    for i in range(size):
        total = cm[i,:]
        recalls[i] =  cm[i][i]/sum(total)
    return recalls

def precisions(cm):
    size = len(cm)
    precisions = np.zeros(size)
    for i in range(size):
        pred_total = cm[:,i]
        precisions[i] = cm[i][i]/sum(pred_total)
    return precisions

def F1Measure(cm):
    avg_recalls = recalls(cm)
    avg_precisions = precisions(cm)
    F1_score = []
    for recall, precision in zip(avg_recalls, avg_precisions):
        F1_score.append( 2*(precision*recall)/(precision+recall))
    return np.array(F1_score)
def classification_rate(cm):
    return sum(np.diag(cm))/sum(sum(cm))
