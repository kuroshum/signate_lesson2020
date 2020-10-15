import numpy as np


values = np.array([10, 3, 1, 5, 8, 6])
print(values)

convert_values = [-1 if values[ind] < 5 else 1 for ind in np.arange(len(values))]
print(convert_values)