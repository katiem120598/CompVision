import numpy as np
from scipy.linalg import svd
from scipy.linalg import lstsq

# Example synthetic data generation (replace with real tracked data)
def generate_synthetic_data():
    F, P, K = 10, 20, 3  # Frames, points, bases
    B_true = np.random.randn(3 * K, P)  # Shape bases
    M_true = np.random.randn(2 * F, 3 * K)  # Scaled rotations
    W = np.dot(M_true, B_true)  # Measurement matrix
    return W, M_true, B_true

# Step 1: Factorization
def factorize_measurements(W, K):
    U, S, Vt = svd(W, full_matrices=False)
    M_tilde = U[:, :3 * K] @ np.diag(S[:3 * K])
    B_tilde = Vt[:3 * K, :]
    return M_tilde, B_tilde

# Step 2: Compute the corrective transformation (G)
def compute_corrective_transformation(M_tilde, B_tilde, F, K):
    # Formulate the constraints (rotation + basis constraints)
    Q = np.zeros((3 * K, 3 * K))  # Corrective transformation constraints
    for f in range(F):
        Mf = M_tilde[2 * f : 2 * f + 2, :]  # Extract rows for each frame
        Q += np.dot(Mf.T, Mf)

    # Solve for G via SVD of Q
    U, _, Vt = svd(Q)
    G = U @ Vt
    return G

# Step 3: Recover M and B
def recover_motion_and_shape(M_tilde, B_tilde, G):
    M = M_tilde @ G
    B = np.linalg.inv(G) @ B_tilde
    return M, B

# Step 4: Main Function
def main():
    # Generate synthetic data
    W, M_true, B_true = generate_synthetic_data()
    F, P = W.shape[0] // 2, W.shape[1]
    K = 3  # Number of bases

    # Factorize the measurement matrix
    M_tilde, B_tilde = factorize_measurements(W, K)

    # Compute the corrective transformation
    G = compute_corrective_transformation(M_tilde, B_tilde, F, K)

    # Recover motion and shape
    M_recovered, B_recovered = recover_motion_and_shape(M_tilde, B_tilde, G)

    # Evaluate reconstruction error
    shape_error = np.linalg.norm(B_true - B_recovered) / np.linalg.norm(B_true)
    motion_error = np.linalg.norm(M_true - M_recovered) / np.linalg.norm(M_true)
    print(f"Shape Reconstruction Error: {shape_error * 100:.2f}%")
    print(f"Motion Reconstruction Error: {motion_error * 100:.2f}%")

if __name__ == "__main__":
    main()
