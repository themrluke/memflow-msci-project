import pandas as pd
import matplotlib.pyplot as plt

def plot_inference_time(csv_file_paths, labels, colors=None):
    """
    Reads inference timing data from multiple CSV files and plots inference time vs batch size with error bars.

    Parameters:
        csv_file_paths (list of str): List of paths to the CSV files containing the timing data.
        labels (list of str): List of labels for each dataset (for legend).
        colors (list of str, optional): List of colors for each dataset (default None uses Matplotlib defaults).
    """
    if len(csv_file_paths) != len(labels):
        raise ValueError("The number of CSV files and labels must be the same.")

    if colors and len(colors) < len(csv_file_paths):
        raise ValueError("Number of colors must be at least equal to the number of CSV files.")

    plt.figure(figsize=(8, 6))

    for i, csv_file_path in enumerate(csv_file_paths):
        try:
            # Load data
            df = pd.read_csv(csv_file_path)

            # Group by batch size, compute mean and standard deviation
            grouped = df.groupby("Batch Size")["Time Taken (seconds)"]
            avg_times = grouped.mean()
            std_dev = grouped.std()  # Standard deviation for error bars

            # Define color (use user-specified or Matplotlib default)
            color = colors[i] if colors else None

            # Plot results with error bars
            plt.errorbar(avg_times.index, avg_times.values, yerr=std_dev.values, 
                         fmt='o-', label=labels[i], capsize=5, color=color)

        except FileNotFoundError:
            print(f"Error: File '{csv_file_path}' not found.")
        except Exception as e:
            print(f"An error occurred with file {csv_file_path}: {e}")

    # Plot formatting
    plt.xlabel("Batch Size")
    plt.ylabel("Time Taken (seconds)")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("Inference Time vs Batch Size (Multiple Models)")
    plt.grid(True)
    plt.legend()
    plt.show()

# For standalone execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python plot_timings.py <csv_file1> <csv_file2> <csv_file3>")
    else:
        csv_files = sys.argv[1:4]
        labels = ["Model 1", "Model 2", "Model 3"]  # Change these labels accordingly
        colors = ["red", "blue", "green"]  # Custom colors for the first three models
        plot_inference_time(csv_files, labels, colors)
