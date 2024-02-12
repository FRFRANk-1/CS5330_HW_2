#include "multiple_histogram_matching.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include "calculateSSD.h"

multipleHistogramMatch::multipleHistogramMatch(std::string f, float s) : filename(std::move(f)), distance(s) {}

cv::Mat calculateHistogram(const cv::Mat& image, int bins) {
    // Compute histogram for a grayscale image or a single channel
    cv::Mat hist;
    int histSize[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    int channels[] = {0};
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    return hist;
}

float compareHistograms(const cv::Mat& histA, const cv::Mat& histB, int method) {
    // Compare histograms using the specified method
    return cv::compareHist(histA, histB, method);
}

void performMultipleHistogramMatching(const std::string& target_multiple_path, const std::string& image_database_dir, int number_of_output) {
    // Load the target image
    cv::Mat target_image = cv::imread(target_multiple_path, cv::IMREAD_COLOR);
    if (target_image.empty()) {
        throw std::runtime_error("Failed to load target image.");
    }
    
    cv::Mat hist_rgb = calculateHistogram(target_image, 8); //  8 bins

    std::vector<multipleHistogramMatch> matches;

    for (const auto& entry : std::filesystem::directory_iterator(image_database_dir)) {
        cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (image.empty()) continue;
        cv::Mat hist_image = calculateHistogram(image, 8); // Calculate histogram for each image
        float distance = calculateSSD(hist_rgb, hist_image); // Calculate SSD or any other distance
        matches.emplace_back(entry.path().filename().string(), distance);
    }

    std::sort(matches.begin(), matches.end(), [](const multipleHistogramMatch& a, const multipleHistogramMatch& b) {
        return a.distance < b.distance;
    });

    // Display top matches
    for (int i = 0; i < std::min(number_of_output, static_cast<int>(matches.size())); i++) {
        std::cout << "Match: " << matches[i].filename << " with distance: " << matches[i].distance << std::endl;
        // Display the matched image using OpenCV window
        cv::Mat matched_image = cv::imread(image_database_dir + "/" + matches[i].filename, cv::IMREAD_COLOR);
        std::string window_name = "Match " + std::to_string(i + 1);
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
        cv::imshow(window_name, matched_image);
    }
    
    // Wait for any key press to close the windows
    cv::waitKey(0);
    cv::destroyAllWindows(); // Close all the opened windows
}
