#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream> // input/output ioerations, such as std::cin, std::cout
#include <filesystem> // for directory operations
#include <algorithm> // for std::sort
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream> 
#include <opencv2/imgproc.hpp>

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

std::vector<std::pair<std::string, std::vector<float>>> computeDataBaseFeatures(const std::string &directoryPath) {
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
    // Compute features for the target image
    cv::Mat targetImage = cv::imread(target_image, cv::IMREAD_GRAYSCALE);
    if (targetImage.empty()) {
        throw std::runtime_error("Failed to open target image: " + target_image);
    }
    std::vector<float> targetFeatures = extractBaseLineFeatures(targetImage);

    // Compute features for all images in the database
    std::vector<std::pair<std::string, std::vector<float>>> databaseFeatures = computeDataBaseFeatures(image_database_dir);

    // Find closest matches
    std::vector<MatchResult> matches = computeAndStoreResults(target_image, databaseFeatures, number_of_output);

    // Write the results to a file using the provided output file path
    std::ofstream outputFile(output_file);
    if (!outputFile.is_open()) {
        throw std::runtime_error("Failed to open output file: " + output_file);
    }
    for (const auto& match : matches) {
        outputFile << match.filename << "," << match.distance << std::endl;
    }
    outputFile.close();
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

cv::Mat computerRGBHistogram(const cv::Mat& image, int binsperchannel = 8) {
    // Split the image into its three channels
    cv::Mat histogram;
    int histsize[] = {binsperchannel, binsperchannel, binsperchannel};
    float range[] = {0, 256};
    const float* ranges[] = {range, range, range};
    int channels[] = {0, 1, 2};

    cv::calcHist(&image, 1, channels, cv::Mat(), histogram, 3, histsize, ranges);
    cv::normalize(histogram, histogram, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());
    return histogram;
}

double histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2) {
    double intersection = 0.0;
    for (int i = 0; i < hist1.rows; ++i) {
        for (int j = 0; j < hist1.cols; ++j) {
            intersection += std::min(hist1.at<float>(i, j), hist2.at<float>(i, j));
        }
    }
    return intersection;
}