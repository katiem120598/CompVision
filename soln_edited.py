import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def create_synthetic_data(num_cameras=3, num_points=50):
    np.random.seed(42)  # For reproducibility
    points_3D = np.random.uniform(-2, 2, (num_points, 3))  # Random 3D points

    # Create cameras looking towards the origin with some noise
    cameras = []
    for _ in range(num_cameras):
        angle = np.random.uniform(0, 2 * np.pi)
        rotation = cv2.Rodrigues(np.array([0, 0, angle]))[0]
        translation = np.random.uniform(-1, 1, 3)
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = rotation
        camera_matrix[:3, 3] = translation
        cameras.append(camera_matrix[:3])

    return points_3D, cameras

def project_points(points_3D, camera_matrix, K):
    # Project points onto the camera
    points_3D_homogeneous = np.hstack((points_3D, np.ones((points_3D.shape[0], 1))))
    projected_points = K @ camera_matrix @ points_3D_homogeneous.T
    projected_points /= projected_points[2]
    return projected_points[:2].T

def plot_3D_structure(cameras, points_3D):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c='b', label='3D Points')

    # Plot camera positions
    for i, cam in enumerate(cameras):
        ax.scatter(cam[0, 3], cam[1, 3], cam[2, 3], c='r', marker='o')
        ax.text(cam[0, 3], cam[1, 3], cam[2, 3], f'Camera {i}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()

def calculate_reprojection_error(points_3D, projected_points, K, cameras):
    total_error = 0
    num_points = points_3D.shape[0]
    for camera_matrix in cameras:
        projections = project_points(points_3D, camera_matrix, K)
        error = np.linalg.norm(projected_points - projections, axis=1).sum()
        total_error += error
    mean_error = total_error / (len(cameras) * num_points)
    return mean_error

# Main execution
num_cameras = 3
num_points = 50
K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]])  # Intrinsic camera matrix
points_3D, cameras = create_synthetic_data(num_cameras, num_points)
projected_points = project_points(points_3D, cameras[0], K)  # Project using the first camera
plot_3D_structure(cameras, points_3D)
error = calculate_reprojection_error(points_3D, projected_points, K, cameras)
print(f"Mean Reprojection Error: {error:.2f} pixels")
