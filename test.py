import pandas as pd
from basis import main as basis_main
from rotation import main as rotation_main
from combo import main as combo_main

def run_iterations(num_iterations=100):
    # Initialize lists to store results
    shape_errors = {"Basis": [], "Rotation": [], "Combo": []}
    motion_errors = {"Basis": [], "Rotation": [], "Combo": []}

    for i in range(1, num_iterations + 1):
        print(f"Running Iteration {i}...")

        # Basis Constraints
        try:
            shape_error, motion_error = basis_main()
            shape_errors["Basis"].append(shape_error)
            motion_errors["Basis"].append(motion_error)
        except Exception as e:
            print(f"Error in Basis Constraints: {e}")
            shape_errors["Basis"].append(None)
            motion_errors["Basis"].append(None)

        # Rotation Constraints
        try:
            shape_error, motion_error = rotation_main()
            shape_errors["Rotation"].append(shape_error)
            motion_errors["Rotation"].append(motion_error)
        except Exception as e:
            print(f"Error in Rotation Constraints: {e}")
            shape_errors["Rotation"].append(None)
            motion_errors["Rotation"].append(None)

        # Combined Constraints
        try:
            shape_error, motion_error = combo_main()
            shape_errors["Combo"].append(shape_error)
            motion_errors["Combo"].append(motion_error)
        except Exception as e:
            print(f"Error in Combo Constraints: {e}")
            shape_errors["Combo"].append(None)
            motion_errors["Combo"].append(None)

    # Create DataFrames for Shape and Motion Errors
    shape_errors_df = pd.DataFrame(shape_errors, index=range(1, num_iterations + 1))
    motion_errors_df = pd.DataFrame(motion_errors, index=range(1, num_iterations + 1))
    
    return shape_errors_df, motion_errors_df

def main():
    # Run iterations and collect results
    num_iterations = 100
    shape_errors_df, motion_errors_df = run_iterations(num_iterations)

    # Display results tables
    print("\nShape Reconstruction Errors:")
    print(shape_errors_df)

    print("\nMotion Reconstruction Errors:")
    print(motion_errors_df)

    # Save to CSV for further analysis (optional)
    shape_errors_df.to_csv("shape_errors.csv", index_label="Iteration")
    motion_errors_df.to_csv("motion_errors.csv", index_label="Iteration")

if __name__ == "__main__":
    main()
