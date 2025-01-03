import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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

# Step 1: Factorization with normalization
def factorize_measurements(W, K):
    # Normalize W to reduce scale inconsistencies
    W_normalized = W / np.linalg.norm(W)
    
    U, S, Vt = svd(W_normalized, full_matrices=False)
    M_tilde = U[:, :3 * K] @ np.diag(S[:3 * K])
    B_tilde = Vt[:3 * K, :]
    
    return M_tilde, B_tilde

# Step 2: Compute the corrective transformation (G) with orthonormalization
### PRIMARY PLACE WHERE BASIS AND ROTATION CONSTRAINTS DIFFER
def compute_corrective_transformation_rotation_constraints(M_tilde, F, K):
    Q = np.zeros((3 * K, 3 * K))  # Corrective transformation constraints
    for f in range(F):
        Mf = M_tilde[2 * f : 2 * f + 2, :]  # Extract rows for each frame
        Q += np.dot(Mf.T, Mf)

    # Solve for G via SVD of Q
    U, _, Vt = svd(Q)
    G = U @ Vt  # Ensure G is orthonormal
    
    # Normalize G
    G /= np.linalg.norm(G)
    
    return G

# Step 3: Recover M and B
def recover_motion_and_shape(M_tilde, B_tilde, G):
    M = M_tilde @ G
    B = np.linalg.inv(G) @ B_tilde
    return M, B

# Visualization: 3D shape reconstruction
def visualize_3d_shapes(B_recovered, B_true=None, frame_idx=0):
    P = B_recovered.shape[1]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    recovered_points = B_recovered[:3, :]
    ax.scatter(recovered_points[0, :], recovered_points[1, :], recovered_points[2, :], c='r', label="Reconstructed")

    if B_true is not None:
        true_points = B_true[:3, :]
        ax.scatter(true_points[0, :], true_points[1, :], true_points[2, :], c='b', label="Ground Truth")

    ax.set_title(f"3D Shape Reconstruction - Frame {frame_idx}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# Visualization: Camera motion trajectory
def visualize_camera_motion(M_recovered, num_frames):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    trajectory = []
    for f in range(num_frames):
        motion_vector = M_recovered[2 * f: 2 * f + 2, :].mean(axis=0)
        trajectory.append(motion_vector)

    trajectory = np.array(trajectory)
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='g', marker='o', label="Camera Trajectory")

    ax.set_title("Camera Motion Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.show()

# Visualization: Reconstruction errors
def plot_reconstruction_error(shape_error, motion_error):
    labels = ["Shape Error", "Motion Error"]
    errors = [shape_error * 100, motion_error * 100]

    plt.figure()
    plt.bar(labels, errors, color=['red', 'blue'])
    plt.title("Reconstruction Errors")
    plt.ylabel("Relative Error (%)")
    plt.grid()
    plt.show()

# Visualization: Animate shape reconstruction
def animate_shapes(B_recovered, B_true=None):
    P = B_recovered.shape[1]
    num_frames = B_recovered.shape[0] // 3

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        recovered_points = B_recovered[frame * 3: (frame + 1) * 3, :]
        ax.scatter(recovered_points[0, :], recovered_points[1, :], recovered_points[2, :], c='r', label="Reconstructed")

        if B_true is not None:
            true_points = B_true[frame * 3: (frame + 1) * 3, :]
            ax.scatter(true_points[0, :], true_points[1, :], true_points[2, :], c='b', label="Ground Truth")

        ax.set_title(f"Frame {frame}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

    ani = FuncAnimation(fig, update, frames=num_frames, repeat=False)
    plt.show()

# Main function
def main(W, M_true, B_true):
    F, P = W.shape[0] // 2, W.shape[1]
    K = 3  # Number of bases

    # Factorize the measurement matrix
    M_tilde, B_tilde = factorize_measurements(W, K)

    # Compute the corrective transformation using Rotation Constraints
    G = compute_corrective_transformation_rotation_constraints(M_tilde, F, K)

    # Recover motion and shape
    M_recovered, B_recovered = recover_motion_and_shape(M_tilde, B_tilde, G)

    # Evaluate reconstruction error
    shape_error = np.linalg.norm(B_true - B_recovered) / np.linalg.norm(B_true)
    motion_error = np.linalg.norm(M_true - M_recovered) / np.linalg.norm(M_true)
    print(f"Shape Reconstruction Error: {shape_error * 100:.2f}%")
    print(f"Motion Reconstruction Error: {motion_error * 100:.2f}%")
    
    """
    visualize_3d_shapes(B_recovered, B_true, frame_idx=0)
    visualize_camera_motion(M_recovered, F)
    plot_reconstruction_error(shape_error, motion_error)
    animate_shapes(B_recovered, B_true)
    """
    

    return shape_error, motion_error

if __name__ == "__main__":
    # Generate synthetic data
    W, M_true, B_true = generate_synthetic_data()
    main(W, M_true, B_true)
