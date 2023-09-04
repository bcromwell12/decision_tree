import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class Node:
    """Node class for decision tree"""

    feature: str
    value: float
    left: "Node"
    right: "Node"
    label: str


class DecisionTree:
    """
    Decision Tree class, This is written as a pretty standard decision tree with the ability to change the impurity calculation

    """

    def __init__(self, max_depth=10, min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root = None

    @staticmethod
    def entropy(y):
        classes = np.unique(y)
        entropy = 0
        for c in classes:
            p_c = len(y[y == c]) / len(y)
            entropy += -p_c * np.log2(p_c)
        return entropy

    @staticmethod
    def gini_index(y):
        classes = np.unique(y)
        gini = 0
        for c in classes:
            p_c = len(y[y == c]) / len(y)
            gini += p_c**2
        return 1 - gini

    def build(self, x_data, y_data, curr_depth=0):
        """
        Build decision tree
        """
        num_samples, num_features = np.shape(x_data)

        if num_samples >= self.min_size and curr_depth <= self.max_depth:
            best_split = self.get_best_split(x_data, num_features)
            if best_split["info_gain"] > 0:
                left_subtree = self.build(best_split["dataset_left"], curr_depth + 1)
                right_subtree = self.build(best_split["dataset_right"], curr_depth + 1)
                return Node(
                    best_split["feature_index"],
                    best_split["threshold"],
                    left_subtree,
                    right_subtree,
                    best_split["info_gain"],
                )
        leaf_value = self.calculate_leaf_value(y_data)
        return Node(value=leaf_value)

    def get_best_split(self, x_data, num_features):
        """function to find the best split"""

        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")

        # loop over all the features
        for feature_index in range(num_features):
            feature_values = x_data[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(x_data, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = x_data[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "gini")
                    # update the best split if needed
                    if curr_info_gain > max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain

        # return best split
        return best_split

    @staticmethod
    def split(dataset, feature_index, threshold):
        """function to split the data"""
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        """Information gain is the process where we determine which set of child nodes improve upon the information gain
        of the parent node."""
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        if mode == "entropy":
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        else:
            gain = self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        return gain

    def calculate_leaf_value(self, y):
        """function to calculate leaf value"""
        classes, counts = np.unique(y, return_counts=True)
        idx = counts.argmax()
        return classes[idx]

    def fit(self, x_data, y_data):
        """fit the model"""
        self.root = self.build(x_data, y_data)

    def predict(self, x_data):
        """simple prediction on the data here"""
        predictions = [self.make_prediction(x, self.root) for x in x_data]
        return predictions
