#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream> // input/output ioerations, such as std::cin, std::cout
#include <filesystem> // for directory operations
#include <algorithm> // for std::sort

#include "process_Img_funs.h"



// Compute features on target Image(Ft)
std :: vector <float> extractBaseLineFeatures(const cv::Mat &image) {
    // Define the patch to extract
    int center_x = image.cols / 2;
    int center_y = image.rows / 2;
    int patch_size = 7;
    // Define region of interest (ROI) for the patch
    cv::Rect roi(center_x - patch_size / 2, center_y - patch_size / 2, patch_size, patch_size);
    // Extract patch from the image
    cv::Mat patch = image(roi);

    // Clone the patch to ensure it's continuous
    cv::Mat continuousPatch = patch.clone(); // This creates a deep copy of the patch

    // Flatten the continuous patch into a 1D vector
    continuousPatch = continuousPatch.reshape(1, 1); // Reshape to 1 row
    std::vector<float> feature;
    continuousPatch.convertTo(feature, CV_32F); // Convert to float and store in vector

    return feature;
}

// Compute features for all image in Database
std::vector<std::pair<std::string, std::vector<float>>> computeDataBaseFeatures(const std::string &directoryPath) {
    std::vector<std::pair<std::string, std::vector<float>>> databaseFeatures;
    //Iterate through files in the directory
    for (const auto &entry : std::filesystem::directory_iterator(directoryPath)) {
        // Extract filename from the path
        std::string filename = entry.path().filename().string();
        // Read image
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        if (!image.empty()) {
            std::vector<float> features = extractBaseLineFeatures(image);
            databaseFeatures.push_back(std::make_pair(filename, features));
        }
    }
    return databaseFeatures;
}

// calculate distance from target to all database image (D(Ft,fi))  
float calculateSSDDistance(const std :: vector<float> &feature1, const std :: vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        throw std :: runtime_error("Feature vectors need to be same size to calculate SSD");
    }

    float sum = 0.0f;
    for (size_t i = 0; i < feature1.size(); i++) {
        sum += (feature1[i] - feature2[i]) * (feature1[i] - feature2[i]);
    }
    return sum;
}

std::vector<std::pair<std::string, float>> findClosestMatches(
    const std::vector<float> &targetFeatures,
    const std::vector<std::pair<std::string, std::vector<float>>> &databaseFeatures,
    int numberOfOutput) {
    
    std::vector<std::pair<std::string, float>> distances;

    for (const auto &pair : databaseFeatures) {
        const auto &filename = pair.first;
        const auto &features = pair.second;
        float distance = calculateSSDDistance(targetFeatures, features);
        distances.emplace_back(filename, distance);
    }

    // Sort the distances
    std::sort(distances.begin(), distances.end(), [](const auto &a, const auto &b) {
        return a.second < b.second;
    });

    // Trim the list to the desired number of outputs
    if (distances.size() > static_cast<size_t>(numberOfOutput)) {
        distances.resize(numberOfOutput);
    }
    return distances;
}