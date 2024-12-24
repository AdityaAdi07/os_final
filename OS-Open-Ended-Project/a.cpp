#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <windows.h>
#include <cstring> // Include for std::memcpy

// Structure to hold pixel data
struct Pixel {
    unsigned char b, g, r; // BMP stores pixel data in BGR format
};

// Function to generate a BMP image
bool generateBMP(const std::string& filename, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing." << std::endl;
        return false;
    }

    // BMP file header
    unsigned int fileSize = 54 + 3 * width * height + height * ((4 - (width * 3) % 4) % 4); // Account for padding
    unsigned char fileHeader[14] = {
        'B', 'M',               // Signature
        0, 0, 0, 0,             // File size (to be filled later)
        0, 0, 0, 0,             // Reserved
        54, 0, 0, 0             // Data offset
    };
    std::memcpy(&fileHeader[2], &fileSize, sizeof(fileSize));

    // BMP info header
    unsigned char infoHeader[40] = {
        40, 0, 0, 0,           
        0, 0, 0, 0,            
        0, 0, 0, 0,            
        1, 0,                  
        24, 0,                  
        0, 0, 0, 0,            
        0, 0, 0, 0,             
        0, 0, 0, 0,             
        0, 0, 0, 0,            
        0, 0, 0, 0,            
        0, 0, 0, 0              
    };
    std::memcpy(&infoHeader[4], &width, sizeof(width));
    std::memcpy(&infoHeader[8], &height, sizeof(height));

    // Write headers to file
    file.write(reinterpret_cast<const char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(infoHeader), sizeof(infoHeader));

    // Generate pixel data (e.g., gradient)
    unsigned char padding[3] = {0, 0, 0};
    int rowPadding = (4 - (width * 3) % 4) % 4;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            Pixel pixel;
            pixel.r = static_cast<unsigned char>((x * 255) / width);
            pixel.g = static_cast<unsigned char>((y * 255) / height);
            pixel.b = static_cast<unsigned char>(128); // Constant value for blue
            file.write(reinterpret_cast<const char*>(&pixel), sizeof(Pixel));
        }
        file.write(reinterpret_cast<const char*>(padding), rowPadding);
    }

    return true;
}

void applyGrayscale(std::vector<Pixel>& pixels, int start, int end) {
    for (int i = start; i < end; ++i) {
        unsigned char gray = static_cast<unsigned char>(0.3 * pixels[i].r + 0.59 * pixels[i].g + 0.11 * pixels[i].b);
        pixels[i].r = pixels[i].g = pixels[i].b = gray;
    }
}

bool processBMP(const std::string& inputFilename, const std::string& outputFilename) {
    std::ifstream file(inputFilename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open input file." << std::endl;
        return false;
    }

    unsigned char fileHeader[14];
    unsigned char infoHeader[40];
    file.read(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.read(reinterpret_cast<char*>(infoHeader), sizeof(infoHeader));

    if (fileHeader[0] != 'B' || fileHeader[1] != 'M') {
        std::cerr << "Error: Not a valid BMP file." << std::endl;
        return false;
    }

    int width = 0, height = 0;
    std::memcpy(&width, &infoHeader[4], sizeof(int));
    std::memcpy(&height, &infoHeader[8], sizeof(int));
     unsigned char padding[3] = {0, 0, 0};
    int rowPadding = (4 - (width * 3) % 4) % 4;

    std::vector<Pixel> pixels(width * height);
    for (int y = 0; y < height; ++y) {
        file.read(reinterpret_cast<char*>(&pixels[y * width]), width * sizeof(Pixel));
        file.ignore(rowPadding);
    }
    file.close();

    int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 4;

    std::vector<std::thread> threads;
    int pixelsPerThread = (width * height) / numThreads;

    for (int i = 0; i < numThreads; ++i) {
        int start = i * pixelsPerThread;
        int end = (i == numThreads - 1) ? (width * height) : (start + pixelsPerThread);
        threads.emplace_back(applyGrayscale, std::ref(pixels), start, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    std::ofstream outFile(outputFilename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Cannot open output file." << std::endl;
        return false;
    }

    outFile.write(reinterpret_cast<const char*>(fileHeader), sizeof(fileHeader));
    outFile.write(reinterpret_cast<const char*>(infoHeader), sizeof(infoHeader));
    for (int y = 0; y < height; ++y) {
        outFile.write(reinterpret_cast<const char*>(&pixels[y * width]), width * sizeof(Pixel));
        outFile.write(reinterpret_cast<const char*>(padding), rowPadding);
    }
    outFile.close();

    return true;
}

int main() {
    std::string generatedFilename = "generated.bmp";
    std::string outputFilename = "processed.bmp";
    int width = 256;
    int height = 256;

    if (!generateBMP(generatedFilename, width, height)) {
        return 1;
    }

    std::cout << "BMP image generated and saved to " << generatedFilename << std::endl;

    if (!processBMP(generatedFilename, outputFilename)) {
        return 1;
    }

    std::cout << "BMP image processed and saved to " << outputFilename << std::endl;
    return 0;
}
