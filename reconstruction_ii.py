import numpy as np
from scipy.linalg import svd, diagsvd

def generate_synthetic_data(num_points, num_frames, num_bases):
    shapes = np.random.rand(3, num_points, num_bases)
    rotations = np.random.randn(num_frames, 2, 3)
    translations = np.random.randn(num_frames, 2, 1)
    measurements = np.zeros((2 * num_frames, num_points))
    for i in range(num_frames):
        for j in range(num_bases):
            projected = rotations[i] @ shapes[:, :, j] + translations[i]
            measurements[2*i:2*i+2, :] += projected
    return measurements, shapes, rotations, translations

def apply_factorization(W, num_bases):
    U, s, Vt = svd(W, full_matrices=False)
    S = diagsvd(s[:3*num_bases], 3*num_bases, 3*num_bases)
    M = U[:, :3*num_bases] @ np.sqrt(S)
    B = np.sqrt(S) @ Vt[:3*num_bases, :]
    return M, B

def calculate_error(original, reconstructed):
    original = original.reshape(-1, original.shape[-1])
    reconstructed = reconstructed.reshape(-1, reconstructed.shape[-1])
    error = np.linalg.norm(original - reconstructed, ord='fro')
    return (error / np.linalg.norm(original, ord='fro')) * 100

def main():
    num_points = 10
    num_frames = 15
    num_bases = 1  # Set to 1 for correct reshaping in this example

    W, true_shapes, true_rotations, true_translations = generate_synthetic_data(num_points, num_frames, num_bases)
    M, B = apply_factorization(W, num_bases)

    if num_bases == 1:
        estimated_rotations = M.reshape(num_frames, 2, 3)
    else:
        raise ValueError("Cannot reshape M as (num_frames, 2, 3) unless num_bases is 1")

    shape_error = calculate_error(true_shapes.reshape(3, -1), B.T.reshape(3, -1))
    print(f"Shape Error: {shape_error:.2f}%")

if __name__ == "__main__":
    main()
