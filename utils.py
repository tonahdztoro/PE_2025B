import numpy as np
import matplotlib.pyplot as plt

def generate_numbers(n, min_val, max_val, median, mean, std, mean_tol, std_tol, max_iter=1000):
    if min_val >= max_val:
        raise ValueError("min_val must be less than max_val")
    if not (min_val <= median <= max_val):
        raise ValueError("median must be between min_val and max_val")
    if n < 2:
        raise ValueError("n must be at least 2")

    arr = np.zeros(n)
    arr[0] = min_val
    arr[-1] = max_val

    # Function to ensure elements are in order and distinct
    def ensure_order_and_distinct(arr):
        for i in range(1, n):
            if arr[i] <= arr[i-1]:
                arr[i] = arr[i-1] + 1e-8  # Ensure distinct and ordered
        return arr

    # Initialize array based on parity of n
    if n % 2 == 1:
        middle = n // 2
        arr[middle] = median
        # Fill the rest with linearly spaced values
        # Before median
        num_before = middle - 1
        if num_before > 0:
            arr[1:middle] = np.linspace(min_val, median, num_before + 2)[1:-1]
        # After median
        num_after = n - middle - 1
        if num_after > 0:
            arr[middle+1:-1] = np.linspace(median, max_val, num_after + 2)[1:-1]
    else:
        middle_left = n // 2 - 1
        middle_right = n // 2
        # Initial delta
        delta = (max_val - min_val) * 0.01
        arr[middle_left] = median - delta
        arr[middle_right] = median + delta
        # Fill before middle_left
        num_before = middle_left - 1
        if num_before > 0:
            arr[1:middle_left] = np.linspace(min_val, median - delta, num_before + 2)[1:-1]
        # Fill after middle_right
        num_after = n - middle_right - 1
        if num_after > 0:
            # Calculate the number of elements to generate
            num_elements = num_after + 2
            # Ensure the slice size matches the number of elements
            slice_length = len(arr[middle_right + 1:-1])
            arr[middle_right + 1:-1] = np.linspace(median + delta, max_val, slice_length + 2)[1:-1]

    # Ensure initial order and distinct
    arr = ensure_order_and_distinct(arr)

    current_mean = np.mean(arr)
    current_std = np.std(arr, ddof=0)
    best_arr = arr.copy()
    best_error = (abs(current_mean - mean) + abs(current_std - std))

    iter_count = 0
    while iter_count < max_iter:
        if abs(current_mean - mean) <= mean_tol and abs(current_std - std) <= std_tol:
            break

        # Perturb the array
        new_arr = np.copy(arr)
        if n % 2 == 0:
            # Adjust delta for even n
            middle_left = n // 2 - 1
            middle_right = n // 2
            current_delta = (new_arr[middle_right] - new_arr[middle_left]) / 2
            delta_change = np.random.normal(0, 0.01 * (max_val - min_val))
            new_delta = current_delta + delta_change
            if new_delta > 0:
                new_ml = median - new_delta
                new_mr = median + new_delta
                if new_ml >= new_arr[middle_left - 1] and new_mr <= new_arr[middle_right + 1]:
                    new_arr[middle_left] = new_ml
                    new_arr[middle_right] = new_mr

        # Adjust other variable positions
        if n % 2 == 1:
            variable_indices = list(range(1, n//2)) + list(range(n//2 + 1, n-1))
        else:
            variable_indices = list(range(1, (n//2 -1))) + list(range((n//2) + 1, n-1))

        for i in variable_indices:
            if i >= len(new_arr):
                continue  # Prevent index error
            perturbation = np.random.normal(0, 0.01 * (max_val - min_val))
            new_val = new_arr[i] + perturbation
            # Ensure new_val is within neighbors
            lower = new_arr[i-1] if i > 0 else min_val
            upper = new_arr[i+1] if i < n-1 else max_val
            new_val = np.clip(new_val, lower + 1e-8, upper - 1e-8)
            new_arr[i] = new_val

        new_arr = ensure_order_and_distinct(new_arr)
        new_mean = np.mean(new_arr)
        new_std = np.std(new_arr, ddof=0)
        new_error = (abs(new_mean - mean) + abs(new_std - std))

        # Keep the better solution
        if new_error < best_error:
            arr = new_arr
            current_mean = new_mean
            current_std = new_std
            best_error = new_error
            best_arr = np.copy(arr)
        iter_count += 1

    # Check final array
    arr = best_arr
    current_mean = np.mean(arr)
    current_std = np.std(arr, ddof=0)
    if abs(current_mean - mean) <= mean_tol and abs(current_std - std) <= std_tol:
        return ensure_order_and_distinct(arr).tolist()
    else:
        raise ValueError(f"Could not converge within {max_iter} iterations. Last mean: {current_mean:.5f}, target: {mean}. Last std: {current_std:.5f}, target: {std}.")
    


def plot_histogram_with_stats(numbers, bins=10):
    # Calculate statistics
    min_val = np.min(numbers)
    max_val = np.max(numbers)
    mean_val = np.mean(numbers)
    median_val = np.median(numbers)
    std_val = np.std(numbers)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot the histogram
    plt.hist(numbers, bins=bins, edgecolor='black', alpha=0.7, label='Data Distribution')
    
    # Plot vertical lines for min, max, mean, median, and std
    plt.axvline(x=min_val, color='r', linestyle='--', label=f'Min: {min_val:.2f}')
    plt.axvline(x=max_val, color='g', linestyle='--', label=f'Max: {max_val:.2f}')
    plt.axvline(x=mean_val, color='m', linestyle='--', label=f'Mean: {mean_val:.2f}')
    plt.axvline(x=median_val, color='c', linestyle='--', label=f'Median: {median_val:.2f}')
    plt.axvline(x=mean_val + std_val, color='y', linestyle=':', label=f'Mean ± Std: {mean_val:.2f} ± {std_val:.2f}')
    plt.axvline(x=mean_val - std_val, color='y', linestyle=':')
    
    # Add labels, title, and legend
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Numbers with Min, Max, Mean, Median, and Std')
    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.6),
          ncol=1, fancybox=True, shadow=True)
    
    # Show the plot
    plt.show()