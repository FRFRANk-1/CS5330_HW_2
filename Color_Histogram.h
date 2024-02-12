#ifndef COLOR_HISTOGRAM_H
#define COLOR_HISTOGRAM_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct ColorHistogramMatchResult {
    std::string filename;
    float similarity;
    
    ColorHistogramMatchResult(std::string f, float s);
};

cv::Mat calcHistForImage(const cv::Mat& image, int bins);
float compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2, int method);
void Color_Histogram_Matching(const std::string& target_histogram_image, const std::string& image_database_dir, const std::string& comparisonMethodStr, int number_of_output);

#endif // COLOR_HISTOGRAM_H