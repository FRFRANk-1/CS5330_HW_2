#ifndef HISTOGRAM_MATCHING_H
#define HISTOGRAM_MATCHING_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct HistogramMatchResult {
    std::string filename;
    float similarity;
    
    HistogramMatchResult(std::string f, float s);
};

enum HistogramComparisonMethod {
    CORRELATION = cv::HISTCMP_CORREL,
    CHI_SQUARE = cv::HISTCMP_CHISQR,
    INTERSECTION = cv::HISTCMP_INTERSECT,
    BHATTACHARYYA = cv::HISTCMP_BHATTACHARYYA
};

cv::Mat calcHistForImage(const cv::Mat& image, int bins);
float compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2, HistogramComparisonMethod method);
void Histogram_Matching(const std::string& target_histogram_image, const std::string& image_database_dir, const std::string& comparisonMethodStr, int number_of_output);

#endif // HISTOGRAM_MATCHING_H
