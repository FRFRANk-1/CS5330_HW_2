#ifndef BASELINE_MATCHING_H
#define BASELINE_MATCHING_H

#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

struct MatchResult {
    std::string filename;
    float distance;

    // Default constructor
    MatchResult() : filename(""), distance(0.0f) {}

    // Constructor with parameters
    MatchResult(std::string fn, float dist) : filename(std::move(fn)), distance(dist) {}

    // Comparator for sorting
    bool operator<(const MatchResult& rhs) const {
        return distance < rhs.distance;
    }
};
// Function declarations
std::vector<float> extractBaseLineFeatures(const cv::Mat& image);
std::vector<std::pair<std::string, std::vector<float>>> computeDataBaseFeatures(const std::string& directoryPath);
std::vector<MatchResult> computeAndStoreResults(const std::string& target_image_path, const std::vector<std::pair<std::string, std::vector<float>>>& databaseFeatures, int number_of_output);
void computeAndStoreResultsAndWriteToFile(const std::string& target_image, const std::string& image_database_dir, int number_of_output, const std::string& output_file);
float calculateSSDDistance(const std::vector<float>& feature1, const std::vector<float>& feature2);

// Function to print feature comparisons
void printFeatureComparisons(const std::string& target_image_path, const std::vector<std::pair<std::string, std::vector<float>>>& databaseFeatures);

// Function to display top matches
void displayTopMatches(const std::vector<MatchResult>& matches, const std::string& image_database_dir, int number_of_top_matches);

// In baseline_matching.h
void performBaselineMatchingAndDisplayResults(const std::string& target_image_path, const std::string& image_database_dir, int number_of_output, const std::string& output_file);

#endif // BASELINE_MATCHING_H
