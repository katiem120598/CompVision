import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes

# Constants
num_frames = 10
num_points = 5
num_bases = 3

# Generate synthetic 3D shape bases as the ground truth
B_true = np.random.randn(3, num_points)  # Using 3xP to maintain 3D shape

# Generate synthetic 2D rotations and translations projected from a 3D to 2D perspective
rotations_true = np.random.randn(num_frames, 2, 3)
translations_true = np.random.randn(num_frames, 2, 1)  # 2D translations

# Initialize arrays for storing errors
motion_errors = []
shape_errors = []

# Noise level
noise_level = 0.1  # Adjust as needed

for f in range(num_frames):
    rotation = rotations_true[f]
    translation = translations_true[f]
    
    # Project the 3D shape points to 2D using the true rotation matrix and add translation
    W_true = rotation @ B_true + translation

    # Introduce noise to the rotation matrix (if noise_level > 0)
    noisy_rotation = rotation + noise_level * np.random.randn(*rotation.shape)
    
    # Project the 3D shape points to 2D using the noisy rotation matrix and add the same translation
    W_estimated = noisy_rotation @ B_true + translation

    # Orthogonal Procrustes to find the best fit rotation that aligns estimated to true
    R_est, scale = orthogonal_procrustes(W_estimated.T, W_true.T)
    W_aligned = R_est @ W_estimated

    # Motion error as the norm of the difference in projected points (2D)
    motion_error = np.linalg.norm(W_true - W_aligned, 'fro')
    motion_errors.append(motion_error)

    # Adjust R_est for 3D (extend R_est from 2x2 to 3x3 by adding a row for the z-axis to maintain 3D integrity)
    R_est_3D = np.eye(3)
    R_est_3D[:2, :2] = R_est
    B_estimated = R_est_3D @ B_true  # Apply estimated rotation to the original 3D points
    
    # Shape error in 3D
    shape_error = np.linalg.norm(B_true - B_estimated, 'fro')
    shape_errors.append(shape_error)

# Calculate average errors
average_motion_error = np.mean(motion_errors)
average_shape_error = np.mean(shape_errors)

# Print errors
print(f"Average Motion Error: {average_motion_error:.4f}")
print(f"Average Shape Error: {average_shape_error:.4f}")

# Plot the results for the first frame
f = 0
plt.figure(figsize=(8, 6))
plt.scatter(W_true[0, :], W_true[1, :], c='blue', label='Ground Truth')
plt.scatter(W_estimated[0, :], W_estimated[1, :], c='red', label='Noisy Data')
plt.scatter(W_aligned[0, :], W_aligned[1, :], c='green', label='Aligned Data')
plt.title(f'Frame {f+1}: Rotation Constraint Method with Translation')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.show()
