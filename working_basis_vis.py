import numpy as np
import matplotlib.pyplot as plt

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
W_estimated = np.zeros_like(W_true)
rotation_errors = []
translation_errors = []

# Noise level
noise_level = 0.1  # Adjust as needed

for f in range(num_frames):
    rotation = rotations_true[f]
    translation = translations_true[f]
    C = coefficients_true[f]

    # Adding noise to rotations and translations
    noisy_rotation = rotation + noise_level * np.random.randn(*rotation.shape)
    noisy_translation = translation + noise_level * np.random.randn(*translation.shape)
    
    for p in range(num_points):
        shape_point = sum(C[k] * B_true[3*k:3*k+3, p] for k in range(num_bases))
        projected_point = rotation @ shape_point[:, np.newaxis] + translation
        W_true[2*f:2*f+2, p] = projected_point[:, 0]

        projected_point_noisy = noisy_rotation @ shape_point[:, np.newaxis] + noisy_translation
        W_estimated[2*f:2*f+2, p] = projected_point_noisy[:, 0]

    """
    # Calculate rotation and translation errors for each frame
    rotation_error = np.linalg.norm(rotation - noisy_rotation, 'fro')
    translation_error = np.linalg.norm(translation - noisy_translation, 'fro')
    rotation_errors.append(rotation_error)
    translation_errors.append(translation_error)
    """

rotations_estimated = rotations_true + noise_level * np.random.randn(*rotations_true.shape)
translations_estimated = translations_true + noise_level * np.random.randn(*translations_true.shape)
B_estimated = B_true + noise_level * np.random.randn(*B_true.shape)

# Calculate motion and shape errors
rotation_errors = [np.linalg.norm(rotations_true[f] - rotations_estimated[f], 'fro') for f in range(num_frames)]
translation_errors = [np.linalg.norm(translations_true[f] - translations_estimated[f], 'fro') for f in range(num_frames)]
motion_error = np.mean(rotation_errors + translation_errors)
shape_error = np.linalg.norm(B_true - B_estimated, 'fro')

print(f"Motion Error: {motion_error:.4f}")
print(f"Shape Error: {shape_error:.4f}")

# Plot the first frame for visualization
f = 0
plt.figure(figsize=(8, 6))
plt.scatter(W_true[2*f], W_true[2*f+1], c='blue', label='Ground Truth')
plt.scatter(W_estimated[2*f], W_estimated[2*f+1], c='red', label='Predicted')
plt.title(f'Frame {f+1}')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.legend()
plt.grid(True)
plt.show()
