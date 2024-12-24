#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <windows.h>
#include <cstring>
#include <chrono>
#include <iomanip>
#include <SFML/Graphics.hpp>

// Structs for Pixel and Processing Metrics
struct Pixel {
    unsigned char b, g, r;
};

struct ProcessingMetrics {
    int threadCount;
    double processingTime;
    int imageWidth;
    int imageHeight;
};

// Graph Visualizer Class
class GraphVisualizer {
private:
    sf::RenderWindow window;
    std::vector<ProcessingMetrics> metrics;
    std::vector<std::vector<int>> colorHistogram;
    std::vector<std::vector<int>> grayHistogram;
    sf::Font font;

public:
    GraphVisualizer() : window(sf::VideoMode(1200, 800), "Image Processing Visualization") {
        if (!font.loadFromFile("arial.ttf")) {
            std::cerr << "Error loading font." << std::endl;
            exit(1);
        }
        window.setFramerateLimit(60);
    }

    void updateMetrics(const ProcessingMetrics& metric) {
        metrics.push_back(metric);
        drawGraphs();
    }

    void updateHistograms(const std::vector<std::vector<int>>& color, const std::vector<std::vector<int>>& gray) {
        colorHistogram = color;
        grayHistogram = gray;
        drawGraphs();
    }

private:
    void drawPerformanceGraph() {
        if (metrics.empty()) return;

        double maxTime = 0;
        for (const auto& metric : metrics) {
            maxTime = std::max(maxTime, metric.processingTime);
        }

        sf::RectangleShape xAxis(sf::Vector2f(500, 2));
        sf::RectangleShape yAxis(sf::Vector2f(2, 300));
        xAxis.setPosition(50, 350);
        yAxis.setPosition(50, 50);
        window.draw(xAxis);
        window.draw(yAxis);

        for (size_t i = 0; i < metrics.size(); ++i) {
            float x = 50 + (i * 100);
            float y = 350 - (metrics[i].processingTime / maxTime * 300);

            sf::CircleShape point(4);
            point.setFillColor(sf::Color::Blue);
            point.setPosition(x - 4, y - 4);
            window.draw(point);

            if (i > 0) {
                float prevX = 50 + ((i - 1) * 100);
                float prevY = 350 - (metrics[i - 1].processingTime / maxTime * 300);
                sf::Vertex line[] = {
                    sf::Vertex(sf::Vector2f(prevX, prevY)),
                    sf::Vertex(sf::Vector2f(x, y))
                };
                window.draw(line, 2, sf::Lines);
            }

            sf::Text label;
            label.setFont(font);
            label.setCharacterSize(12);
            label.setFillColor(sf::Color::White);
            label.setString(std::to_string(metrics[i].threadCount) + " threads");
            label.setPosition(x - 20, 360);
            window.draw(label);
        }

        sf::Text title;
        title.setFont(font);
        title.setString("Processing Time vs Thread Count");
        title.setCharacterSize(20);
        title.setFillColor(sf::Color::White);
        title.setPosition(150, 20);
        window.draw(title);
    }

    void drawHistogram(const std::vector<std::vector<int>>& histogram, float xOffset, const std::string& title) {
        if (histogram.empty()) return;

        int maxVal = 0;
        for (const auto& channel : histogram) {
            for (int val : channel) {
                maxVal = std::max(maxVal, val);
            }
        }

        sf::Color colors[] = {sf::Color::Red, sf::Color::Green, sf::Color::Blue};
        for (int channel = 0; channel < 3; ++channel) {
            for (int i = 0; i < 256; ++i) {
                float height = (histogram[channel][i] / static_cast<float>(maxVal)) * 200;
                sf::RectangleShape bar(sf::Vector2f(1, height));
                bar.setPosition(xOffset + i, 750 - height);
                bar.setFillColor(colors[channel]);
                window.draw(bar);
            }
        }

        sf::Text titleText;
        titleText.setFont(font);
        titleText.setString(title);
        titleText.setCharacterSize(20);
        titleText.setFillColor(sf::Color::White);
        titleText.setPosition(xOffset + 50, 450);
        window.draw(titleText);
    }

    void drawGraphs() {
        window.clear(sf::Color(30, 30, 30));
        drawPerformanceGraph();
        drawHistogram(colorHistogram, 50.f, "Color Histogram");
        drawHistogram(grayHistogram, 600.f, "Grayscale Histogram");
        window.display();
    }
};

// Histogram Calculation
std::vector<std::vector<int>> calculateHistogram(const std::vector<Pixel>& pixels) {
    std::vector<std::vector<int>> histogram(3, std::vector<int>(256, 0));
    for (const auto& pixel : pixels) {
        histogram[0][pixel.r]++;
        histogram[1][pixel.g]++;
        histogram[2][pixel.b]++;
    }
    return histogram;
}

// Multi-Threaded BMP Processing
double processBMPWithThreads(const std::vector<Pixel>& inputPixels,
                             std::vector<Pixel>& outputPixels, int threadCount) {
    auto startTime = std::chrono::high_resolution_clock::now();

    int size = inputPixels.size();
    auto worker = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            int gray = (inputPixels[i].r + inputPixels[i].g + inputPixels[i].b) / 3;
            outputPixels[i] = {static_cast<unsigned char>(gray),
                               static_cast<unsigned char>(gray),
                               static_cast<unsigned char>(gray)};
        }
    };

    std::vector<std::thread> threads;
    int chunkSize = size / threadCount;
    for (int t = 0; t < threadCount; ++t) {
        int start = t * chunkSize;
        int end = (t == threadCount - 1) ? size : start + chunkSize;
        threads.emplace_back(worker, start, end);
    }
    for (auto& th : threads) th.join();

    auto endTime = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = endTime - startTime;
    return duration.count();
}

// BMP Generation
bool generateBMP(const std::string& filename, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    unsigned char fileHeader[14] = {
        'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0
    };
    unsigned char infoHeader[40] = {40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0};

    int fileSize = 54 + width * height * 3;
    std::memcpy(&fileHeader[2], &fileSize, sizeof(int));
    std::memcpy(&infoHeader[4], &width, sizeof(int));
    std::memcpy(&infoHeader[8], &height, sizeof(int));

    file.write(reinterpret_cast<const char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<const char*>(infoHeader), sizeof(infoHeader));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char color[3] = {255, 255, 255};
            file.write(reinterpret_cast<const char*>(color), 3);
        }
    }
    file.close();
    return true;
}

int main() {
    GraphVisualizer visualizer;

    std::string generatedFilename = "generated.bmp";
    std::string outputFilename = "processed.bmp";
    int width = 1024, height = 1024;

    if (!generateBMP(generatedFilename, width, height)) {
        std::cerr << "Failed to generate BMP image." << std::endl;
        return 1;
    }

    std::ifstream inputFile(generatedFilename, std::ios::binary);
    if (!inputFile.is_open()) {
        std::cerr << "Error: Cannot open generated BMP file." << std::endl;
        return 1;
    }

    std::vector<Pixel> inputPixels(width * height);
    inputFile.seekg(54);
    inputFile.read(reinterpret_cast<char*>(&inputPixels[0]), width * height * sizeof(Pixel));
    inputFile.close();

    std::vector<Pixel> outputPixels = inputPixels;
    std::vector<int> threadCounts = {1, 2, 4, 8, 16};
    for (int threadCount : threadCounts) {
        double processingTime = processBMPWithThreads(inputPixels, outputPixels, threadCount);
        visualizer.updateMetrics({threadCount, processingTime, width, height});
    }

    auto colorHist = calculateHistogram(inputPixels);
    auto grayHist = calculateHistogram(outputPixels);
    visualizer.updateHistograms(colorHist, grayHist);

    // Main Visualization Loop
    while (true) {
        sf::Event event;
        while (visualizer.window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                visualizer.window.close();
                return 0;
            }
        }
    }

    return 0;
}
