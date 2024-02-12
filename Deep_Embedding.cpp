#include "deep_embedding.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/highgui.hpp>
#include <cmath> // For cosine calculation

std::vector<ImageEmbedding> results;


void DeepEmbedding::loadEmbeddings() {
    std::ifstream file(csvFilePath);
    std::string line;
    while (std::getline(file, line)) {
        // Split the line by comma
        std::stringstream ss(line);
        // First entry is the filename  
        std::string filename;
        // First entry is the filename
        std::getline(ss, filename, ',');
        
        std::vector<float> embedding;
        std::string value;
        while (std::getline(ss, value, ',')) {
            // Convert the string to a float and add it to the embedding vector
            embedding.push_back(std::stof(value));
        }
        // Add the filename and embedding to the embeddings vector
        embeddings.emplace_back(filename, embedding);
    }
}

std::vector<ImageEmbedding> DeepEmbedding::findNearestNeighbors(const std::string& targetFilename, int topK, const std::string& distanceMetric) const {
    std::vector<float> targetEmbedding;
    for(const auto& embedding : embeddings) {
        if(embedding.filename == targetFilename) {
            targetEmbedding = embedding.embedding;
            break;
        }
    }

    // Check if the target embedding was found
    if (targetEmbedding.empty()) {
        std::cerr << "Target embedding not found for filename: " << targetFilename << std::endl;
        return {}; // Return an empty vector
    }

    // Create a temporary vector for distances
    std::vector<std::pair<float, std::string>> distances;

    for(const auto& embedding : embeddings) {
        if(embedding.filename != targetFilename) {
            float distance = 0.0f;
            if (distanceMetric == "sum-square") {
                distance = calculateSumSquareDistance(targetEmbedding, embedding.embedding);
            } else if (distanceMetric == "cosine") {
                distance = calculateCosineDistance(targetEmbedding, embedding.embedding);
            }
            distances.emplace_back(distance, embedding.filename);
        }
    }

    // Sort based on distance
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first < b.first;
    });

    // Extract the top K results
    std::vector<ImageEmbedding> nearestNeighbors;
    for (int i = 0; i < std::min(topK, static_cast<int>(distances.size())); ++i) {
        auto& embeddingPair = distances[i];
        for (const auto& embedding : embeddings) {
            if (embedding.filename == embeddingPair.second) {
                nearestNeighbors.push_back(embedding);
                break;
            }
        }
    }

    return nearestNeighbors;
}

void DeepEmbedding::displayResults(const std::vector<ImageEmbedding>& results) const {
    for (const auto& result : results) {
        std::cout << "Match: " << result.filename << std::endl;
        std::string imagePath = imageDatabaseDir + "/" + result.filename; // Make sure imageDatabaseDir is correctly set up in your class
        cv::Mat img = cv::imread(imagePath);
        if (!img.empty()) {
            cv::imshow("Match", img);
            cv::waitKey(0); // Wait for any key press
        } else {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
        }
    }
    cv::destroyAllWindows();
}


float DeepEmbedding::calculateCosineDistance(const std::vector<float>& v1, const std::vector<float>& v2) const {
    // Placeholder for cosine distance calculation
    float dotProduct = 0.0f;
    float normV1 = 0.0f;
    float normV2 = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        dotProduct += v1[i] * v2[i];
        normV1 += v1[i] * v1[i];
        normV2 += v2[i] * v2[i];
    }

    normV1 = std::sqrt(normV1);
    normV2 = std::sqrt(normV2);
    return 1.0f - dotProduct / (normV1 * normV2);
}

float DeepEmbedding::calculateSumSquareDistance(const std::vector<float>& v1, const std::vector<float>& v2) const {
    float distance = 0.0f;
    for (size_t i = 0; i < v1.size(); i++) {
        distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return distance;
}

void Deep_Embedding(const std::string& targetImageFilename, const std::string& csvFilePath, const std::string& imageDatabaseDir, int number_of_outputs) {
    DeepEmbedding deepEmbedding(csvFilePath, imageDatabaseDir);
    deepEmbedding.loadEmbeddings();
    auto results = deepEmbedding.findNearestNeighbors(targetImageFilename, number_of_outputs, "cosine");
    deepEmbedding.displayResults(results);
}

