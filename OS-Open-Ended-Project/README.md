# Efficient Multithreaded Image Processing

## Project Description
This project demonstrates an efficient approach to image processing using multithreading. It involves generating an image, applying grayscale conversion using parallel processing, and saving the processed output. The project highlights the performance benefits of multithreading in handling computationally intensive tasks like image manipulation.

## Key Features
- **Image Generation**: Dynamically generates a color gradient image.
- **Grayscale Conversion**: Converts the generated image to grayscale using multithreaded processing.
- **Customizable Parameters**: Allows modification of image resolution for flexibility.
- **Thread Optimization**: Distributes pixel processing across multiple threads for improved performance.

## Significance
Image processing is a critical component in various domains such as:
- **Satellite Imagery**: Enhancing image clarity, analyzing terrains, or detecting anomalies.
- **Medical Imaging**: Processing X-rays, MRIs, and CT scans for diagnosis.
- **Security Applications**: Filtering and analyzing images for threat detection.

This project showcases how parallel computing can optimize image processing for such use cases.

## How It Works
### 1. Image Generation
A color gradient image is generated programmatically. Each pixel is assigned values based on its position to form a smooth gradient.

### 2. Grayscale Conversion
Grayscale conversion is performed using a weighted average formula:
```text
Gray = 0.3 * R + 0.59 * G + 0.11 * B
```
The image is divided into segments, and each thread processes a specific segment of pixels.

### 3. Multithreading
The workload is distributed among available threads to accelerate processing. For instance:
- Thread 1 processes rows 0–249.
- Thread 2 processes rows 250–499.
- This pattern continues based on the resolution and the number of threads.

## Usage Instructions
### Prerequisites
- **Operating System**: Windows (required for Windows-specific APIs)
- **Compiler**: GCC/MinGW or any C++ compiler with multithreading support.

### Steps to Run
1. Clone the repository.
2. Open the project folder in your development environment, remember to use latest g++ compiler (64bit).
3. Compile the code:
   ```bash
   g++ -o image_processing project.cpp -std=c++11 -lpthread
   ```
4. Run the executable:
   ```bash
   ./image_processing
   ```
5. The program generates and processes the image, saving the output files in the current directory.

### Modifying Resolution
To change the image resolution:
1. Open the code file.
2. Update the `width` and `height` variables in the `main()` function. For example:
   ```cpp
   int width = 1024;
   int height = 1024;
   ```
3. Recompile and run the program.

## File Structure
- **project.cpp**: Contains the main code.
- **generated_image.bmp**: The dynamically generated image.
- **processed_image.bmp**: The grayscale image after processing.

## Performance Insights
- Multithreading significantly reduces processing time by dividing the workload across CPU cores.
- For larger resolutions (e.g., 1024×1024), the performance improvement is more noticeable.

## Future Enhancements
- **Support for Additional Filters**: Add edge detection, sharpening, or blurring.
- **Image Input Support**: Enable users to load custom images instead of generating them.
- **Cross-Platform Compatibility**: Extend support for Linux and macOS.
- **Dynamic Thread Allocation**: Adjust the number of threads dynamically based on the system's capabilities.

## Conclusion
This project illustrates how multithreaded programming can enhance image processing efficiency. It serves as a foundational implementation for exploring advanced parallel processing techniques in image manipulation.

---
### Author
Anirudh C  & Aditya Ankanath
**Bachelor’s in Information Science Engineering**  
R V College of Engineering, Bengaluru

---
Feel free to reach out for suggestions or improvements!

