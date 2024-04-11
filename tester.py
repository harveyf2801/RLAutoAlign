import numpy as np

# Assuming array1 and array2 are your two numpy arrays
array1 = np.array([[1, 2], [3, 4]])
array2 = np.array([[5, 6], [7, 8]])

# Flatten the arrays
flattened_array1 = array1.flatten()
flattened_array2 = array2.flatten()

# Concatenate the flattened arrays
combined_array = np.concatenate((flattened_array1, flattened_array2))

print(combined_array)