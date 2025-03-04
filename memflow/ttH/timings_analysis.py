import pandas as pd
import matplotlib.pyplot as plt

def plot_inference_time(csv_file_path):
    """
    Reads inference timing data from a CSV file and plots inference time vs batch size.

    Parameters:
        csv_file_path (str): Path to the CSV file containing the timing data.
    """
    try:
        # Load data
        df = pd.read_csv(csv_file_path)

        # Group by batch size and compute average time taken
        avg_times = df.groupby("Batch Size")["Time Taken (seconds)"].mean()

        # Plot results
        plt.figure(figsize=(8, 6))
        plt.plot(avg_times.index, avg_times.values, marker='o', linestyle='-')
        plt.xlabel("Batch Size")
        plt.ylabel("Time Taken (seconds)")
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Inference Time vs Batch Size")
        plt.grid(True)
        plt.show()

    except FileNotFoundError:
        print(f"Error: File '{csv_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# For standalone execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python plot_timings.py <csv_file_path>")
    else:
        plot_inference_time(sys.argv[1])
