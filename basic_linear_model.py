# -*- coding: utf-8 -*-
"""
Created on Fri May  4 17:32:53 2018

@author: J Wolf
"""
import numpy as np


class BasicLinearModel:
    def __init__(self, splits=None):
        if splits is None:
            # NOTE: The marathon race splits default is every 5 kilometers plus half and finish
            splits = [5, 10, 15, 20, 21.0975, 25, 30, 35, 40, 42.195]
        self.splits = np.array(splits)

    def predict(self, X):

        last_split = X.shape[1] - 1
        pace = (
            X[:, 0] / self.splits[0]
            if last_split == 0
            else (X[:, -1] - X[:, -2])
            / (self.splits[last_split] - self.splits[last_split - 1])
        )
        pace = pace.reshape(X.shape[0], 1)

        current = X[:, -1].reshape(X.shape[0], 1)

        predictions = np.full(
            (X.shape[0], len(self.splits) - last_split - 1),
            self.splits[last_split + 1 :] - self.splits[last_split],
        )
        predictions = pace * predictions + current

        return predictions
