import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
import psutil  

# Function to calculate MSE
def mean_squared_error(image1, image2):
    # Ensure both images are grayscale (2D)
    if image1.ndim == 3:  # Convert to grayscale if the image is in color (3 channels)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if image2.ndim == 3:  # Convert to grayscale if the image is in color (3 channels)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute MSE
    return np.sum((image1 - image2) ** 2) / float(image1.shape[0] * image1.shape[1])

# Function to compare images using SSIM
def compare_images(image1, image2):
    # Ensure images are grayscale for SSIM
    if image1.ndim == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if image2.ndim == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    return ssim(image1, image2)

# Grayscale conversion function (single-threaded)
def to_grayscale_single(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Grayscale conversion function (multi-threaded)
def to_grayscale_threaded(image, num_threads=4):
    def process_chunk(start, end):
        return cv2.cvtColor(image[start:end, :], cv2.COLOR_BGR2GRAY)

    # Split image into chunks and process in parallel
    height = image.shape[0]
    chunk_size = height // num_threads
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_threads)]
    
    # Handle last chunk with remaining rows
    if height % num_threads != 0:
        chunks[-1] = (chunks[-1][0], height)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(lambda chunk: process_chunk(chunk[0], chunk[1]), chunks)

    # Combine results
    result = np.vstack(list(results))
    return result

# Function to track CPU usage
def track_cpu_usage():
    # Get current CPU usage percentage
    return psutil.cpu_percent(interval=0.1)  # Interval of 0.1 sec to get accurate reading

# Performance comparison function
def performance_comparison(image, num_threads=4):
    # Track CPU usage before single-threaded processing
    cpu_usage_before_single = track_cpu_usage()

    start_time = time.time()
    grayscale_single = to_grayscale_single(image)
    single_time = time.time() - start_time

    # Track CPU usage after single-threaded processing
    cpu_usage_after_single = track_cpu_usage()

    # Track CPU usage before multi-threaded processing
    cpu_usage_before_threaded = track_cpu_usage()

    start_time = time.time()
    grayscale_threaded = to_grayscale_threaded(image, num_threads)
    threaded_time = time.time() - start_time

    # Track CPU usage after multi-threaded processing
    cpu_usage_after_threaded = track_cpu_usage()

    # Calculate average CPU usage during both processes
    cpu_usage_single = (cpu_usage_before_single + cpu_usage_after_single) / 2
    cpu_usage_threaded = (cpu_usage_before_threaded + cpu_usage_after_threaded) / 2
    
    return single_time, threaded_time, cpu_usage_single, cpu_usage_threaded

# Main function to process and compare images
def main():
    # Load the image
    image = cv2.imread('generated.bmp')  # Make sure to use the appropriate image path
    
    if image is None:
        print("Error: Could not load image.")
        return
    
    # Grayscale conversion using single-threaded
    grayscale_single = to_grayscale_single(image)
    
    # Grayscale conversion using multi-threaded
    grayscale_threaded = to_grayscale_threaded(image)
    
    # Compare grayscale and color images using MSE and SSIM
    mse_value = mean_squared_error(grayscale_single, image)
    ssim_value = compare_images(grayscale_single, image)
    
    print(f'Mean Squared Error (MSE) between grayscale and color image: {mse_value}')
    print(f'SSIM between grayscale and color image: {ssim_value}')
    
    # Display images for comparison
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 3, 2)
    plt.title("Grayscale Image")
    plt.imshow(grayscale_single, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title("Grayscale Image (Threaded)")
    plt.imshow(grayscale_threaded, cmap='gray')
    
    plt.show()
    
    # Performance comparison: Threaded vs Non-threaded
    single_time, threaded_time, cpu_usage_single, cpu_usage_threaded = performance_comparison(image)
    
    print(f"Multi-threaded processing time: {single_time:.4f} seconds")
    print(f"Single-threaded processing time: {threaded_time:.4f} seconds")
    print(f"CPU usage (Single-threaded): {cpu_usage_single:.2f}%")
    print(f"CPU usage (Multi-threaded): {cpu_usage_threaded:.2f}%")
    
    # Plot performance comparison (Time)
    labels = ['Multi-threaded', 'Single-threaded']
    times = [single_time, threaded_time]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(labels, times, color=['blue', 'green'])
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison: Threaded vs Non-threaded')
    
    # Plot CPU usage comparison
    labels = ['Multi-threaded', 'Single-threaded']
    cpu_usages = [cpu_usage_threaded, cpu_usage_single]
    plt.subplot(1, 2, 2)
    plt.bar(labels, cpu_usages, color=['blue', 'green'])
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Comparison: Threaded vs Non-threaded')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
