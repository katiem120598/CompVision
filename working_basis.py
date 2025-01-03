import numpy as np

# Constants
num_frames = 10
num_points = 5
num_bases = 3

# Generate synthetic shape bases (3K x P matrix) as the ground truth
B_true = np.random.randn(3*num_bases, num_points)

# Generate synthetic rotations and translations as the ground truth
rotations_true = np.random.randn(num_frames, 2, 3)
translations_true = np.random.randn(num_frames, 2, 1)

# Generate synthetic coefficients for combinations
coefficients_true = np.random.randn(num_frames, num_bases)

# Form the full measurement matrix W as ground truth (2F x P)
W_true = np.zeros((2*num_frames, num_points))
for f in range(num_frames):
    rotation = rotations_true[f]
    translation = translations_true[f]
    C = coefficients_true[f]
    
    for p in range(num_points):
        shape_point = sum(C[k] * B_true[3*k:3*k+3, p] for k in range(num_bases))
        projected_point = rotation @ shape_point[:, np.newaxis] + translation
        W_true[2*f:2*f+2, p] = projected_point[:, 0]

# Simulate errors by adding noise
noise_level = 0.0  # Adjust noise level to simulate estimation error
rotations_estimated = rotations_true + noise_level * np.random.randn(*rotations_true.shape)
translations_estimated = translations_true + noise_level * np.random.randn(*translations_true.shape)
B_estimated = B_true + noise_level * np.random.randn(*B_true.shape)

# Calculate motion and shape errors
rotation_errors = [np.linalg.norm(rotations_true[f] - rotations_estimated[f], 'fro') for f in range(num_frames)]
translation_errors = [np.linalg.norm(translations_true[f] - translations_estimated[f], 'fro') for f in range(num_frames)]
motion_error = np.mean(rotation_errors + translation_errors)
shape_error = np.linalg.norm(B_true - B_estimated, 'fro')

print(f"Motion Error: {motion_error:.2f}")
print(f"Shape Error: {shape_error:.2f}")
