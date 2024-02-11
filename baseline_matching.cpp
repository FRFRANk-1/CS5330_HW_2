#include "baseline_matching.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <utility>

std::vector<float> extractBaseLineFeatures(const cv::Mat& image) {
    int center_x = image.cols / 2;
    int center_y = image.rows / 2;
    int patch_size = 7;
    cv::Rect roi(center_x - patch_size / 2, center_y - patch_size / 2, patch_size, patch_size);
    cv::Mat patch = image(roi).clone().reshape(1, 1);
    std::vector<float> feature;
    patch.convertTo(feature, CV_32F);
    return feature;
}

std::vector<std::pair<std::string, std::vector<float>>> computeDataBaseFeatures(const std::string& directoryPath) {
    std::vector<std::pair<std::string, std::vector<float>>> databaseFeatures;
    for (const auto& entry : std::filesystem::directory_iterator(directoryPath)) {
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            std::vector<float> features = extractBaseLineFeatures(image);
            databaseFeatures.push_back({entry.path().filename().string(), features});
        }
    }
    return databaseFeatures;
}

std::vector<MatchResult> computeAndStoreResults(const std::string& target_image_path, const std::vector<std::pair<std::string, std::vector<float>>>& databaseFeatures, int number_of_output) {
    cv::Mat targetImage = cv::imread(target_image_path, cv::IMREAD_GRAYSCALE);
    auto targetFeatures = extractBaseLineFeatures(targetImage);
    std::vector<MatchResult> results;

    for (const auto& [filename, features] : databaseFeatures) {
        float distance = calculateSSDDistance(targetFeatures, features);
        results.emplace_back(filename, distance);
    }

    std::sort(results.begin(), results.end());
    if (results.size() > number_of_output) {
        results.resize(number_of_output);
    }
    return results;
}

void computeAndStoreResultsAndWriteToFile(const std::string& target_image, const std::string& image_database_dir, int number_of_output, const std::string& output_file) {
    std::vector<std::pair<std::string, std::vector<float>>> databaseFeatures = computeDataBaseFeatures(image_database_dir);
    std::vector<MatchResult> matches = computeAndStoreResults(target_image, databaseFeatures, number_of_output);
    
    std::ofstream outputFile(output_file);
    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }
    for (const auto& match : matches) {
        outputFile << match.filename << "," << match.distance << std::endl;
    }
    outputFile.close();
}

float calculateSSDDistance(const std::vector<float>& feature1, const std::vector<float>& feature2) {
    if (feature1.size() != feature2.size()) {
        throw std::runtime_error("Feature vectors need to be the same size to calculate SSD");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < feature1.size(); i++) {
        sum += (feature1[i] - feature2[i]) * (feature1[i] - feature2[i]);
    }
    return sum;
}
