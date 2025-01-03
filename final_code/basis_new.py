import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_symmetric_shape_bases(P, K):
    """
    Create the synthetic data following symmetric constraint
    """
    half_P = P // 2  # Number of points on one side of the face
    B_symmetric = np.random.randn(3 * K, half_P)  # Generate random data for one side

    # Create symmetric shape by mirroring the left side to the right
    B_full = np.concatenate((B_symmetric, np.flip(B_symmetric, axis=1)), axis=1)

    return B_full

def generate_synthetic_data(F, P, K):
    """
    Generate synthetic data with symmetric shape bases.
    """
    B_true = generate_symmetric_shape_bases(P, K)
    M_true = np.random.randn(2 * F, 3 * K)  # Scaled rotations/motions
    W = np.dot(M_true, B_true)  # Simulated observed data matrix
    return W, M_true, B_true

def factorize_measurements(W, K):
    # Normalize and factorize measurements
    W_normalized = W / np.linalg.norm(W)
    U, S, Vt = svd(W_normalized, full_matrices=False)
    return U[:, :3 * K] @ np.diag(S[:3 * K]), Vt[:3 * K, :]

def recover_motion_and_shape(M_tilde, B_tilde, K):
    # Apply an additional SVD to ensure the motion and shape matrices are well-formed
    G, _, _ = svd(np.dot(M_tilde.T, M_tilde))
    M = M_tilde @ G[:3 * K, :]
    B = np.linalg.inv(G[:3 * K, :]) @ B_tilde
    return M, B

def visualize_3d_shapes(B, title="3D Shape"):
    # Visualize the 3D shape points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(B[0, :], B[1, :], B[2, :], c='r', marker='o')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def main():
    F, P, K = 10, 20, 3  # Frames, points, shape bases; ensure P is even
    W, M_true, B_true = generate_synthetic_data(F, P, K)
    M_tilde, B_tilde = factorize_measurements(W, K)
    M_recovered, B_recovered = recover_motion_and_shape(M_tilde, B_tilde, K)

    # Evaluate reconstruction errors
    shape_error = np.linalg.norm(B_true - B_recovered) / np.linalg.norm(B_true)
    motion_error = np.linalg.norm(M_true - M_recovered) / np.linalg.norm(M_true)
    
    print(f"Shape Reconstruction Error: {shape_error * 100:.2f}%")
    print(f"Motion Reconstruction Error: {motion_error * 100:.2f}%")

    # Visualization of the original and reconstructed shapes
    #visualize_3d_shapes(B_true, title="Original Symmetric Shape")
    #visualize_3d_shapes(B_recovered, title="Reconstructed Shape")

    return shape_error, motion_error

if __name__ == "__main__":
    main()
