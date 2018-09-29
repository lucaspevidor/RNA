import csv
import os
import numpy as np

print(os.listdir())
with open('./RNA/TP3/dataset_teste.data', newline='') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    x = np.empty([1, 123])
    for row in reader:
        print(row)
        x = np.append(x, np.array([np.array(row)]), axis=0)
