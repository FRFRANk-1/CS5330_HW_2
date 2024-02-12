#include "texture_color.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include "calculateSSD.h"



cv::Mat CalColorHistogram(const cv::Mat& image, int bins) {
    if (image.empty()) {
        std::cerr << "Failed to load image for color histogram calculation." << std::endl;
        return cv::Mat(); // Return an empty matrix to indicate failure
    }

    int histSize[] = {bins}; // Number of bins for each channel
    float hranges[] = {0, 256}; // Range for histogram values
    const float* ranges[] = {hranges}; // Same range for all channels
    int channels[] = {0, 1, 2}; // Channels for BGR image

    cv::Mat hist;
    cv::MatND histograms[3]; // To store histogram for each channel

    for (int i = 0; i < 3; i++) {
        cv::calcHist(&image, 1, &channels[i], cv::Mat(), // No mask
                     histograms[i], 1, histSize, ranges,
                     true, // Uniform
                     false); // Not accumulate
    }

    // Assuming you want to combine histograms or use them separately
    // For demonstration, let's normalize and return the histogram of the first channel
    cv::normalize(histograms[0], hist, 0, 1, cv::NORM_MINMAX);

    return hist;
}

cv::Mat CalTextureHistogram(const cv::Mat& image, int bins) {
    cv::Mat gradX, gradY;
    cv::Sobel(image, gradX, CV_32F, 1, 0);
    cv::Sobel(image, gradY, CV_32F, 0, 1);

    // Compute magnitude of gradients
    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);

    // Calculate histogram
    cv::Mat textureHist;
    int histSize[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    int channels[] = {0};
    cv::calcHist(&magnitude, 1, channels, cv::Mat(), textureHist, 1, histSize, ranges, true, false);
    
    // Normalize histogram
    cv::normalize(textureHist, textureHist, 0, 1, cv::NORM_MINMAX);
    
    return textureHist;
}

cv::Mat combineHistograms(const cv::Mat& colorHist, const cv::Mat& textureHist) {
    // Normalize histograms to ensure equal contribution
    cv::Mat normalizedColorHist, normalizedTextureHist;
    cv::normalize(colorHist, normalizedColorHist, 0, 1, cv::NORM_MINMAX);
    cv::normalize(textureHist, normalizedTextureHist, 0, 1, cv::NORM_MINMAX);

    // Concatenate histograms into a single feature vector
    cv::Mat combinedHist;
    cv::hconcat(normalizedColorHist, normalizedTextureHist, combinedHist);
    
    return combinedHist;
}

float calculateDistance(const cv::Mat& colorHist1, const cv::Mat& textureHist1, 
                        const cv::Mat& colorHist2, const cv::Mat& textureHist2) {
    // Calculate SSD for texture histograms
    float textureSSD = calculateSSD(textureHist1, textureHist2);

    // Calculate distance using another metric (e.g., correlation or chi-square) for color histograms
    float colorDistance = cv::compareHist(colorHist1, colorHist2, cv::HISTCMP_CORREL); // Example using correlation

    // Average the distances
    return (textureSSD + colorDistance) / 2.0f;
}

void texture_color(const std::string& target_texture_color_image, const std::string& image_database_dir, int number_of_output) {
    // Load the target image
    cv::Mat targetImage = cv::imread(target_texture_color_image);
    if (targetImage.empty()) {
        std::cerr << "Failed to load target image." << std::endl;
        return;
    }

    // Calculate color histogram and texture histogram for the target image
    cv::Mat targetColorHist = CalColorHistogram(targetImage, 8);
    std::cout << "Size of target color histogram: " << targetColorHist.size() << std::endl;

    cv::Mat targetTextureHist = CalTextureHistogram(targetImage, 8);
    std::cout << "Size of target texture histogram: " << targetTextureHist.size() << std::endl;

    // Store distances and filenames
    std::vector<std::pair<float, std::string>> distances;

    // Iterate through the images in the database
    for (const auto& entry : std::filesystem::directory_iterator(image_database_dir)) {
        cv::Mat databaseImage = cv::imread(entry.path().string());
        if (databaseImage.empty()) continue;

        // Calculate color histogram and texture histogram for the database image
        cv::Mat databaseColorHist = CalColorHistogram(databaseImage, 8);
        cv::Mat databaseTextureHist = CalTextureHistogram(databaseImage, 8);

        // Calculate distance metric
        float distance = calculateDistance(targetColorHist, targetTextureHist, databaseColorHist, databaseTextureHist);

        // Store distance and filename
        distances.emplace_back(distance, entry.path().filename().string());
    }

    // Sort distances
    std::sort(distances.begin(), distances.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first;
    });
    
   
    // Display top matches
    for (int i = 0; i < std::min(number_of_output, static_cast<int>(distances.size())); ++i) {
    std::cout << "Match: " << distances[i].second << " with distance: " << distances[i].first << std::endl;

    // Construct the full path to the matched image
    std::string matchedImagePath = image_database_dir + "/" + distances[i].second;

    // Load the matched image
    cv::Mat matchedImage = cv::imread(matchedImagePath);

    // Check if the image is loaded successfully
    if (matchedImage.empty()) {
        std::cerr << "Failed to load image: " << matchedImagePath << std::endl;
        continue; // Skip this iteration if image is not loaded
    }

    // Create a window to display the image
    std::string windowName = "Match " + std::to_string(i + 1) + ": " + distances[i].second;
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);

    // Display the image in the window
    cv::imshow(windowName, matchedImage);
}

    // Wait for any key press to close the windows
    cv::waitKey(0);

    // Destroy all the windows created by your application to free up resources
    cv::destroyAllWindows();
}



