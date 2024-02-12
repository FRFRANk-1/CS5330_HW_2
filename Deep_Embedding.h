#ifndef DEEP_EMBEDDING_H
#define DEEP_EMBEDDING_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

// Represents a single embedding for an image
struct ImageEmbedding {
    std::string filename;
    std::vector<float> embedding;

    ImageEmbedding(std::string filename, std::vector<float> embedding)
        : filename(std::move(filename)), embedding(std::move(embedding)) {}
};

class DeepEmbedding {
public:
    // DeepEmbedding::DeepEmbedding(const std::string& csvFilePath) : csvFilePath(csvFilePath) {}

    DeepEmbedding(const std::string& csvFilePath, const std::string& imageDirPath) 
    : csvFilePath(csvFilePath), imageDatabaseDir(imageDirPath) {}
    void loadEmbeddings();
    std::vector<ImageEmbedding> findNearestNeighbors(const std::string& targetFilename, int topK = 3, const std::string& distanceMetric = "cosine") const;
    void displayResults(const std::vector<ImageEmbedding>& results) const;

private:
    std::vector<ImageEmbedding> embeddings;
    std::string csvFilePath;
    std::string imageDatabaseDir;

    float calculateCosineDistance(const std::vector<float>& v1, const std::vector<float>& v2) const;
    float calculateSumSquareDistance(const std::vector<float>& v1, const std::vector<float>& v2) const;
};

void Deep_Embedding(const std::string& targetImageFilename, const std::string& csvFilePath, const std::string& imageDatabaseDir, int number_of_output);
#endif // DEEP_EMBEDDING_H
