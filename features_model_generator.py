import os
import numpy as np
import pandas as pd
import time
from yearly_data import YearlyData

directory = "features"

if not os.path.exists(directory):
    os.makedirs(directory)

yearly = YearlyData('../dataset/input/train.csv', '../dataset/input/weather.csv', test_csv='../dataset/input/test.csv')
yearly_test = YearlyData('../dataset/input/test.csv', '../dataset/input/weather.csv')

print("Pre-processing train dataset")
data_train = yearly.process()
data_train.to_csv(directory + "/train.csv", index=False)


print("\nPre-processing test dataset")
data_test = yearly_test.process()
data_test.to_csv(directory + "/test.csv", index=False)

print("\nDatasets generated")