import numpy as np
import matplotlib.pyplot as plt

# Constants
num_frames = 10
num_points = 5
num_bases = 1  # Ensure num_bases is 1

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic shape bases (3K x P matrix) as the ground truth
B_true = np.random.randn(3 * num_bases, num_points)

# Generate synthetic rotations and translations as the ground truth
rotations_true = np.random.randn(num_frames, 2, 3)
translations_true = np.random.randn(num_frames, 2, 1)

# Generate synthetic coefficients for combinations
coefficients_true = np.random.randn(num_frames, num_bases)

# Generate synthetic measurements with and without noise
W_true = np.zeros((2 * num_frames, num_points))
W_estimated = np.zeros_like(W_true)
noise_level = 0.0  # Adjust noise level here

for f in range(num_frames):
    R = rotations_true[f]
    t = translations_true[f]
    for p in range(num_points):
        shape_point = sum(coefficients_true[f, k] * B_true[3*k:3*k+3, p] for k in range(num_bases))
        projected_point = R @ shape_point[:, np.newaxis] + t
        W_true[2*f:2*f+2, p] = projected_point[:, 0]

        # Adding Gaussian noise to the projected points for W_estimated
        noise = noise_level * np.random.randn(2, 1)
        W_estimated[2*f:2*f+2, p] = projected_point[:, 0] + noise[:, 0]

# Step 1: Apply SVD to the estimated measurement matrix
U, S, Vt = np.linalg.svd(W_estimated, full_matrices=False)

# Step 2: Estimate the motion and shape matrices
M_estimated = U[:, :3*num_bases] @ np.diag(np.sqrt(S[:3*num_bases]))
B_estimated = np.diag(np.sqrt(S[:3*num_bases])) @ Vt[:3*num_bases, :]

# Step 3: Recover the shapes using the estimated motion and shape matrices
recovered_shapes = np.zeros((3, num_points, num_frames))
for f in range(num_frames):
    R_estimated = M_estimated[2*f:2*f+2, :3]
    t_estimated = np.zeros((2, 1))  # Assuming zero translation for simplicity
    for p in range(num_points):
        # Adjusted shape_point calculation for num_bases=1
        shape_point = B_estimated[:, p] * coefficients_true[f, 0]
        recovered_shapes[:, p, f] = shape_point

# Step 4: Calculate errors
shape_error = np.linalg.norm(B_true - B_estimated) / np.linalg.norm(B_true) * 100
motion_error = np.linalg.norm(rotations_true - M_estimated[:, :3].reshape(num_frames, 2, 3)) / np.linalg.norm(rotations_true) * 100

print(f"Shape Error: {shape_error:.2f}%")
print(f"Motion Error: {motion_error:.2f}%")

# Visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(B_true[0, :], B_true[1, :], B_true[2, :], color='blue', label='Original')
ax.scatter(B_estimated[0, :], B_estimated[1, :], B_estimated[2, :], color='red', alpha=0.5, label='Recovered')
ax.set_title("Shape Recovery")
ax.legend()
plt.show()