import numpy as np
import pandas as pd
from rotation_revised import main as rotation_main
from basis_new import main as basis_main
from basis_new import *
import matplotlib.pyplot as plt

def generate_symmetric_shape_bases(P, K):
    """
    Create the synthetic data following symmetric constraint
    """
    half_P = P // 2  # Number of points on one side of the face
    B_symmetric = np.random.randn(3 * K, half_P)  # Generate random data for one side

    # Create symmetric shape by mirroring the left side to the right
    B_full = np.concatenate((B_symmetric, np.flip(B_symmetric, axis=1)), axis=1)

    return B_full

def generate_synthetic_data_with_noise(F, P, K, noise):
    """
    Generate synthetic data with symmetric shape bases.
    """
    B_true = generate_symmetric_shape_bases(P, K)
    M_true = np.random.randn(2 * F, 3 * K)  # Scaled rotations/motions
    W = np.dot(M_true, B_true)  # Simulated observed data matrix

    # Add Gaussian noise to the measurement matrix
    noise = noise * np.random.randn(*W.shape)
    W_noisy = W + noise

    return W_noisy, M_true, B_true, noise

def run_method_iteration(method, W, M_true, B_true):
    if method == "Basis":
        shape_error, motion_error = basis_main()
    elif method == "Rotation":
        shape_error, motion_error = rotation_main(W, M_true, B_true)
    else:
        raise ValueError("Invalid method specified!")
    return shape_error, motion_error

# iterate through noise levels
def run_iterations_with_noise(num_iterations=100, noise_increment=0.01):
    methods = ["Basis", "Rotation"]
    results = []

    for i in range(0, num_iterations):
        noise_level = i * noise_increment
        print(f"Running Iteration {i}, Noise Level: {noise_level:.3f}")

        # Generate noisy synthetic data
        W, M_true, B_true, _ = generate_synthetic_data_with_noise(10,20,3, noise_level)

        row = {"Iteration": i, "Noise Level": noise_level}
        for method in methods:
            try:
                print(f"Running {method} Method...")
                shape_error, motion_error = run_method_iteration(method, W, M_true, B_true)
                row[f"{method} Shape Error (%)"] = shape_error * 100
                row[f"{method} Motion Error (%)"] = motion_error * 100
            except Exception as e:
                print(f"Error in {method} Method, Iteration {i}: {e}")
                row[f"{method} Shape Error (%)"] = None
                row[f"{method} Motion Error (%)"] = None

        results.append(row)

    results_df = pd.DataFrame(results)
    return results_df

# Statistical and Plotting Functions
def analyze_and_plot(df):
    # get descriptive stats and improvement of basis model over rotation model
    print(df.describe())
    shape_improvement = 100 * (df['Rotation Shape Error (%)'] - df['Basis Shape Error (%)']) / df['Rotation Shape Error (%)']
    motion_improvement = 100 * (df['Rotation Motion Error (%)'] - df['Basis Motion Error (%)']) / df['Rotation Motion Error (%)']
    print(f"Mean Shape Improvement: {shape_improvement.mean():.2f}%")
    print(f"Mean Motion Improvement: {motion_improvement.mean():.2f}%")


    # plot where basis model did better at reconstructing shapes than rotation model
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(df["Noise Level"], df["Basis Shape Error (%)"] < df["Rotation Shape Error (%)"], c='blue', label='Shape Error Comparison')
    plt.title("Noise Levels where Basis Shape Error < Rotation Shape Error")
    plt.xlabel("Noise Level")
    plt.ylabel("Basis Better (True/False)")
    plt.grid(True)

    # plot where basis model did better at reconstructing motion than rotation model
    plt.subplot(1, 2, 2)
    plt.scatter(df["Noise Level"], df["Basis Motion Error (%)"] < df["Rotation Motion Error (%)"], c='red', label='Motion Error Comparison')
    plt.title("Noise Levels where Basis Motion Error < Rotation Motion Error")
    plt.xlabel("Noise Level")
    plt.ylabel("Basis Better (True/False)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Error Distribution Plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df.boxplot(column=['Basis Shape Error (%)', 'Rotation Shape Error (%)'])
    plt.title('Box Plot of Shape Errors')

    plt.subplot(1, 2, 2)
    df.boxplot(column=['Basis Motion Error (%)', 'Rotation Motion Error (%)'])
    plt.title('Box Plot of Motion Errors')
    plt.show()

    # Scatter Plot of Errors
    plt.scatter(df['Basis Shape Error (%)'], df['Basis Motion Error (%)'], alpha=0.5, label='Basis Method')
    plt.scatter(df['Rotation Shape Error (%)'], df['Rotation Motion Error (%)'], alpha=0.5, label='Rotation Method')
    plt.xlabel('Shape Error (%)')
    plt.ylabel('Motion Error (%)')
    plt.legend()
    plt.title('Scatter Plot of Shape vs. Motion Errors')
    plt.show()

# Main function
def main():
    num_iterations = 100
    noise_increment = 0.01
    results_df = run_iterations_with_noise(num_iterations, noise_increment)
    analyze_and_plot(results_df)

    # Save results to a single csv file
    filename = "reconstruction_errors_with_noise.csv"
    results_df.to_csv(filename, index=False)
    #plot shape error
    results_df.plot(x="Noise Level", y=["Basis Shape Error (%)", "Rotation Shape Error (%)"], title="Shape Reconstruction Errors")
    plt.show()
    results_df.plot(x="Noise Level", y=["Basis Motion Error (%)", "Rotation Motion Error (%)"], title="Motion Reconstruction Errors")
    plt.show()

    # show beginning of table
    print("\nResults (First 5 rows):")
    print(results_df.head())

if __name__ == "__main__":
    main()
