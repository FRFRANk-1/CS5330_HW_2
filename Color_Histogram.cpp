#include "color_histogram.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

ColorHistogramMatchResult::ColorHistogramMatchResult(std::string f, float s)
    : filename(std::move(f)), similarity(s) {}

cv::Mat calc_color_Hist_For_Image(const cv::Mat& image, int bins) {
    // Convert the image to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Calculate the histogram for the H, S, and V channels
    int hBins = bins, sBins = bins, vBins = bins;
    int histSize[] = { hBins, sBins, vBins };
    float hRanges[] = { 0, 180 };
    float sRanges[] = { 0, 256 };
    float vRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges, sRanges, vRanges };
    int channels[] = { 0, 1, 2 };

    cv::Mat hist;
    cv::calcHist(&hsvImage, 1, channels, cv::Mat(), hist, 3, histSize, ranges, true, false);

    // Normalize the histogram
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);
    
    return hist;
}

double compare_color_histograms(const cv::Mat& hist1, const cv::Mat& hist2, int method) {
    return cv::compareHist(hist1, hist2, method);
}

void Color_Histogram_Matching(const std::string& target_histogram_image, const std::string& image_database_dir, const std::string& comparisonMethodStr, int number_of_output) {
    // Load target image and calculate its histogram
    cv::Mat targetImage = cv::imread(target_histogram_image);
    cv::Mat targetHist = calcHistForImage(targetImage, 32); // Using 32 bins for illustration

    // Determine comparison method
    int method = (comparisonMethodStr == "intersection") ? cv::HISTCMP_INTERSECT : cv::HISTCMP_CHISQR;

    // Vector to store match results
    std::vector<ColorHistogramMatchResult> matchResults;

    // Iterate over images in the database directory
    for (const auto& entry : std::filesystem::directory_iterator(image_database_dir)) {
        cv::Mat image = cv::imread(entry.path().string());
        cv::Mat hist = calcHistForImage(image, 32);
        float similarity = compareHistograms(targetHist, hist, method);
        matchResults.emplace_back(entry.path().filename().string(), similarity);
    }

    // Sort the results based on similarity
    std::sort(matchResults.begin(), matchResults.end(),
              [](const ColorHistogramMatchResult& a, const ColorHistogramMatchResult& b) {
                  return a.similarity > b.similarity; // For intersection, bigger is better
              });

    // Output the top matching results
    for (int i = 0; i < std::min(number_of_output, static_cast<int>(matchResults.size())); ++i) {
        std::cout << "Match: " << matchResults[i].filename
                  << " with similarity: " << matchResults[i].similarity << std::endl;
    }
}
