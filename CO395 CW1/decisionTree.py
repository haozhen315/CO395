import numpy as np

class DecisionTree():
    def __init__(self,train_set,train_label, level = 1):

        self.x = train_set
        self.y = train_label
        self.n = self.x.shape[0]
        self.col = self.x.shape[1]
        self.leaf = self.is_leaf()
        self.level = level
        self.gain= 0
        self.buildTree()

    def buildTree(self):
        if self.is_leaf():
            self.classes = self.majority_vote()
            return
        else:
            self.find_split()
            l_idx = np.nonzero(self.x[:,self.attribute] <= self.value)[0]
            r_idx = np.nonzero(self.x[:,self.attribute] > self.value)[0]

            if len(l_idx) == 0 or len(r_idx) == 0:
                self.leaf = True
                self.classes = self.majority_vote()
                return
            else:
                self.left = DecisionTree(self.x[l_idx], self.y[l_idx],self.level +1)
                self.right = DecisionTree(self.x[r_idx], self.y[r_idx], self.level + 1)

    def find_split(self):
        for var_idx in range(self.col):
            x = self.x[:,var_idx]
            sort_idx = np.argsort(x)
            sort_y, sort_x = self.y[sort_idx], x[sort_idx]
            curr_label=sort_y[0]

            for i in range(1,self.n):
                #only calculate information gain when there is a label change
                if sort_y[i] != curr_label:
                    curr_label = sort_y[i]
                    s_left = sort_y[:i]
                    s_right = sort_y[i:]
                    curr_gain = self.infomation_gain(sort_y, s_left, s_right)
                    if curr_gain > self.gain:
                        self.gain = curr_gain
                        self.value = (sort_x[i] + sort_x[i-1]) /2
                        self.attribute = var_idx

    def entropy(self, label):
         # Count number of label and frequency
        unique, counts = np.unique(label, return_counts=True)
        dist = counts / sum(counts)
        H = -sum(dist * np.log2(dist))
        return H

    def infomation_gain(self, S_all, S_left, S_right):
        remainder = len(S_left)/(len(S_left)+len(S_right)) * self.entropy(S_left) + len(S_right)/(len(S_left)+len(S_right)) * self.entropy(S_right)
        gain = self.entropy(S_all) - remainder
        return gain

    def majority_vote(self):
        label, counts = np.unique(self.y, return_counts=True)
        return label[np.argmax(counts)]


    def get_depth(self):
        if self.leaf:
            return self.level
        else:
            l_depth=self.left.get_depth()
            r_depth =self.right.get_depth()
            max_depth = max(l_depth, r_depth)
            return max_depth

    def get_num_leafs(self):
        if self.leaf: return 1
        else:
            return self.left.get_num_leafs() + self.right.get_num_leafs()


    def predict(self, test_set):
        prediction = np.array([self.predict_row(xi) for xi in test_set])
        return prediction

    def predict_row(self, x_new):
        if self.leaf:
            return self.classes
        else:
            if x_new[self.attribute] <= self.value:
                return self.left.predict_row(x_new)
            else:
                return self.right.predict_row(x_new)

    def is_leaf(self):
        return len(np.unique(self.y)) == 1

    def accuracy(self, test_set):
        actual = test_set[:,-1]
        prediction = self.predict(test_set[:,:7])
        return sum(actual == prediction)/ len(actual)

    def prune(self,valid_set):
        if self.left.leaf and self.right.leaf:
            if len(valid_set) ==0:
                return
            else:
                bef_acc =self.accuracy(valid_set)
                aft_label = self.majority_vote()
                aft_acc = np.sum(valid_set[:,-1] == aft_label) / valid_set.shape[0]
                if aft_acc > bef_acc:
                    self.leaf = True
                    self.classes = aft_label
                    del self.left
                    del self.right
                    return
        else:
            # divide subset by recursion
            idx = valid_set[:,self.attribute] < self.value
            left_subset_valid=valid_set[idx,:]
            right_subset_valid = valid_set[np.logical_not(idx),:]

            # Three other possible cases with different copy rule
            if self.left.leaf ==False and self.right.leaf == True:
                self.left.prune(left_subset_valid)

            elif self.left.leaf ==True and self.right.leaf == False:
                self.right.prune(right_subset_valid)

            elif self.left.leaf ==False and self.right.leaf == False:
                self.left.prune(left_subset_valid)
                self.right.prune(right_subset_valid)



    def post_prune(self, valid_set):
        '''
        This function take a training set and validation set, build a trained tree and apply the reduce-error-pruning rule on the tree level by level
        until there is no more error reduced.
        Output is the pruned tree.
        '''
        curr_score = self.accuracy(valid_set)
        while True:
            self.prune(valid_set)
            tmp_score = self.accuracy(valid_set)
            if tmp_score <= curr_score:
                break
            else:
                curr_score = tmp_score

    def confusion_matrix(self, valid_set):
        prediction = self.predict(valid_set[:,:7])
        actual = valid_set[:,-1]
        labels = np.unique(actual)
        cm = np.zeros((len(labels), len(labels)),dtype=np.int32)
        for a, p in zip(actual.astype(int), prediction.astype(int)):
            cm[a-1][p-1] += 1
        return cm
