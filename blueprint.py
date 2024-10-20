import numpy as np

# Example training data (features and labels)
training_data = np.array([
    [1.0, 2.0, 0],  # [feature1, feature2, class_label]
    [1.5, 1.8, 0],
    [5.0, 8.0, 1],
    [6.0, 9.0, 1]
])

# New data point to classify
new_data_true = np.array([2.0, 6.0])
new_data_false = np.array([3.0, 3.5])

# Gaussian kernel function
def gaussian_kernel(distance, sigma=1.0):
    return np.exp(-distance**2 / (2 * sigma**2))

# Calculate Gaussian kernel values between new_data_true and each training point
kernel_values_true = []
for data_point in training_data:
    distance_true = np.linalg.norm(new_data_true - data_point[:2])
    kernel_value_true = gaussian_kernel(distance_true)
    kernel_values_true.append((kernel_value_true, data_point[2]))

#repeat for new_data_false
kernel_values_false = []
for data_point in training_data:
    distance_false = np.linalg.norm(new_data_false - data_point[:2])
    kernel_value_false = gaussian_kernel(distance_false)
    kernel_values_false.append((kernel_value_false, data_point[2]))

# Separate kernel values by class
class_1_kernels_true = [kv[0] for kv in kernel_values_true if kv[1] == 0]
class_2_kernels_true = [kv[0] for kv in kernel_values_true if kv[1] == 1]

class_1_kernels_false = [kv[0] for kv in kernel_values_false if kv[1] == 0]
class_2_kernels_false = [kv[0] for kv in kernel_values_false if kv[1] == 1]

# Sum kernel values for each class
class_1_sum_true = sum(class_1_kernels_true)
class_2_sum_true = sum(class_2_kernels_true)

class_1_sum_false = sum(class_1_kernels_false)
class_2_sum_false = sum(class_2_kernels_false)


# Predict the class with the highest sum of kernel values (probability)
predicted_class_true = 0 if class_1_sum_true > class_2_sum_true else 1
predicted_class_false = 0 if class_1_sum_false > class_2_sum_false else 1
print(f"Predicted class (assuming true): {predicted_class_true}")
print(f"Predicted class (assuming false): {predicted_class_false}")