#ifndef MULTIPLE_HISTOGRAM_MATCHING_H
#define MULTIPLE_HISTOGRAM_MATCHING_H

#include <string>
#include <opencv2/core.hpp> // Include necessary OpenCV headers

class multipleHistogramMatch {
public:
    std::string filename;
    float distance;

    multipleHistogramMatch(std::string f, float s);
};

float calculateSSD(const cv::Mat& hist1, const cv::Mat& hist2);

cv::Mat calculateHistogram(const cv::Mat& image, int bins);

float compareHistograms(const cv::Mat& histA, const cv::Mat& histB, int method);

void performMultipleHistogramMatching(const std::string& target_multiple_path, const std::string& image_database_dir, int number_of_output);

#endif // MULTIPLE_HISTOGRAM_MATCHING_H
