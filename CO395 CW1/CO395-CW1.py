
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
from time import time


def entropy(dataset):
     # Count number of label and frequency
    unique, counts = np.unique(dataset, return_counts=True)
    dist = counts / sum(counts)
    H = -sum(dist * np.log2(dist))
    return H

def infomation_gain(S_all, S_left, S_right):
    remainder = S_left.shape[0]/(S_left.shape[0]+S_right.shape[0]) * entropy(S_left) + S_right.shape[0]/(S_left.shape[0]+S_right.shape[0]) * entropy(S_right)
    gain = entropy(S_all) - remainder
    return gain


def Decision_tree_traversal(node):
    tree_dict ={}
    if(type(node) is Node):
        tree_dict["attribute"]= node.attribute
        tree_dict["value"]= node.value

        tree_dict["left"] = Decision_tree_traversal(node.left)

        tree_dict["right"] = Decision_tree_traversal(node.right)
    else:
        tree_dict["label"] = node.label
    return tree_dict

def predict(node, x_new):
    if x_new[node.attribute] < node.value:
        if type(node.left) is Node:
            return predict(node.left, x_new)
        else:
            return node.left.label
    else:
        if type(node.right) is Node:
            return predict(node.right, x_new)
        else:
            return node.right.label


def accuracy(root, data):
    if len(data) ==0:
        return 0
    predictions =[]
    for xi in data:
        prediction = predict(root,xi)
        predictions.append(prediction)
    actual = data[:,-1]
    accuracy = sum(actual == predictions)/len(actual)
    return accuracy


class Tree:
    def __init__(self,root):
        self.root = None

    def decision_tree_learning_main(self, train_dataset, max_depth):
        root = self.decision_tree_learning(train_dataset, max_depth)[0]
        self.root = root

    def decision_tree_learning(self, train_dataset, max_depth, depth=0):
        # if the current depth reaches the depth limit, return the main vote of labels of the current dataset
        max_depth = max_depth
        if depth >= max_depth:
            labels, counts = np.unique(train_set[:, -1], return_counts=True)
            return Leaf(labels[np.argmax(counts)]), depth
        
        # if all samples have the same label, return leaf
        if len(np.unique(train_dataset[:, -1])) == 1:
            return Leaf(train_dataset[0][-1]), depth
        else:
            #node = find_split(train_dataset)
            attribute, value = self.find_split(train_dataset);
            node = Node(attribute, value)

            l_dataset = train_dataset[train_dataset[:, node.attribute] <= node.value]
            r_dataset = train_dataset[train_dataset[:, node.attribute] > node.value]
            if len(l_dataset) == 0 or len(r_dataset) == 0:
                # labels: unique labels
                labels, counts = np.unique(np.concatenate((l_dataset, r_dataset), axis=0)[:, -1], return_counts=True)
                return Leaf(labels[np.argmax(counts)]), depth
            node.left, l_dph= self.decision_tree_learning(l_dataset, max_depth, depth = depth + 1)
            node.right, r_dph= self.decision_tree_learning(r_dataset, max_depth, depth = depth + 1)
            return node, max(l_dph, r_dph)

    def find_split(self, training_dataset):
        y = training_dataset[:,-1]
        idx = np.arange(len(y))
        row = len(idx)
        col  = training_dataset.shape[1] - 1
        tmp = np.zeros([col,2])
        prev_gain = 0
        split = 0
        split_col = 0
        for var_idx in range(col):
            x_col = training_dataset[:,var_idx]
            sort_idx = np.argsort(x_col)
            sort_y, sort_x = y[sort_idx], x_col[sort_idx]
            n = x_col.shape[0]
            y_class=sort_y[0]
            for i in range(1,n):
                #only calculate information gain when there is a label change
                if sort_y[i]!=y_class:

                    y_class=sort_y[i]
                    x_i = sort_x[i]
                    s_left = sort_y[:i-1]
                    s_right = sort_y[i-1:]
                    curr_gain = infomation_gain(sort_y, s_left, s_right)
                    if curr_gain > prev_gain:
                        prev_gain = curr_gain
                        split = x_i
                        split_col = var_idx
        #return ('attribute':split_col, 'value':split)
        return(split_col, split)


class Node:
    def __init__(self,attribute,value):
        self.attribute = attribute #Wi-Fi 1-7
        self.value = value
        self.left = None
        self.right = None

class Leaf:
    def __init__(self,label):
        self.label = label


if __name__== "__main__":
    # Load the clean dataset
    path= 'wifi_db/clean_dataset.txt'
    data = np.loadtxt(path)
    np.random.seed(42)
    train_idx = np.random.choice(2000, 1000, replace = False)
    test_idx = np.delete(np.arange(data.shape[0]), train_idx)
    train_set = data[train_idx, :]
    test_set = data[test_idx, :]
    tree = Tree(None)
    for dep in range(1, 15):
        tree.decision_tree_learning_main(train_set, dep)
#         print("start")
        #print(Decision_tree_traversal(tree.root))
#         print("end")
        print('depth:', dep, "accuracy:", accuracy(tree.root,test_set))

