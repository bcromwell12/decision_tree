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


@dataclass()
class DecisionTree:
    """
    Decision Tree class
    """
    def __init__(self, max_depth=10, min_size=1):
        self.max_depth = max_depth
        self.min_size = min_size

    def build(self, x_data:, y_data:, max_depth=10, min_size=1):
        """
        Build decision tree
        """
        root = self.get_split(x_data, y_data)
        self.split(root, max_depth, min_size, 1)
        return root

