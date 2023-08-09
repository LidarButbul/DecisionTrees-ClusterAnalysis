import numpy as np
from sklearn.linear_model import LogisticRegression

class Node:

    def __init__(self, x, y, min_leaf=5, class_weight=None, max_depth=5, depth=0):
        self.x = [list(s) for s in x]
        self.y = list(y)
        self.min_leaf = min_leaf
        self.num_of_samples = len(y)
        self.prediction = None
        self.features = [i for i in range(len(x[0]))]
        if all(num == 0 for num in y) or all(num == 1 for num in y):
            self.prediction = y[0]
        else:
            self.reg_model = LogisticRegression().fit(x, y)
        self.left_child = None
        self.right_child = None
        self.split = None
        self.max_depth = max_depth

    def grow_tree(self):
        if self.max_depth == 0:
            return
        if self.num_of_samples == 2 * self.min_leaf:
            return
        if all(num == 0 for num in self.y) or all(num == 1 for num in self.y):
            return
        if len(self.x) == 0:
            return
        threshold, gini, feature = None, 0, None
        for i in self.features:
            threshold_i, gini_i = self.find_best_split(i)
            if gini < gini_i:
                gini = gini_i
                threshold = threshold_i
                feature = i
        x_left, x_right, y_left, y_right = [], [], [], []
        if feature is not None:
            for s in self.x:
                if s[feature] <= threshold:
                    x_left.append(s)
                    y_left.append(self.y[self.x.index(s)])
                else:
                    x_right.append(s)
                    y_right.append(self.y[self.x.index(s)])
        if len(x_left) == 0 or len(x_right) == 0:
            return
        self.split = (feature, threshold)
        self.left_child = Node(x_left, y_left, self.min_leaf, max_depth=self.max_depth - 1)
        self.right_child = Node(x_right, y_right, self.min_leaf, max_depth=self.max_depth - 1)
        self.left_child.grow_tree()
        self.right_child.grow_tree()

    def find_best_split(self, var_idx):
        vals = [row[var_idx] for row in self.x]
        gini_gain = 0
        best = None
        for val in np.unique(vals):
            lhs = [vals.index(i) for i in vals if i <= val]
            rhs = [vals.index(i) for i in vals if i > val]
            if len(lhs) < self.min_leaf or len(rhs) < self.min_leaf:
                continue
            val_gini = self.get_gini_gain(lhs, rhs)
            if gini_gain < val_gini:
                gini_gain = val_gini
                best = val
        return best, gini_gain

    def get_gini_gain(self, lhs, rhs):
        y1_l_count, y2_l_count, y1_r_count, y2_r_count = 0, 0, 0, 0
        for x in lhs:
            if self.y[x] == 0:
                y1_l_count += 1
            else:
                y2_l_count += 1
        for x in rhs:
            if self.y[x] == 0:
                y1_r_count += 1
            else:
                y2_r_count += 1
        node_gini = self.gini_impurity(y1_l_count + y1_r_count, y2_l_count + y2_r_count)
        p_l = len(lhs) / (len(lhs) + len(rhs))
        p_r = len(rhs) / (len(lhs) + len(rhs))
        return node_gini - (p_l * self.gini_impurity(y1_l_count, y2_l_count) + p_r * self.gini_impurity(y1_r_count,y2_r_count))

    def is_leaf(self):
        if self.left_child is None and self.right_child is None:
            return True
        return False

    def predict(self, x):
        return [self.predict_row(i) for i in x]

    def predict_row(self, xi):
        if self.is_leaf():
            if self.prediction is None:
                return self.reg_model.predict([xi])[0]
            return self.prediction
        if xi[self.split[0]] <= self.split[1]:
            return self.left_child.predict_row(xi)
        return self.right_child.predict_row(xi)

    @staticmethod
    def gini_impurity(y1_count, y2_count):
        p_1 = y1_count / (y1_count + y2_count)
        p_2 = y2_count / (y1_count + y2_count)
        return 1 - (p_1 ** 2 + p_2 ** 2)
