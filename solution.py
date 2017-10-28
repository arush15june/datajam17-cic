import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn import linear_model, datasets

"""
Datasets
"""
training_set_DF = pandas.read_csv("datasets/training_set.csv")
test_set_DF = pandas.read_csv("datasets/testing_set.csv")

print(training_set_DF)
print(test_set_DF)