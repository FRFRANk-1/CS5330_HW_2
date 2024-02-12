#ifndef CALCULATE_DISTANCE_H
#define CALCULATE_DISTANCE_H

#include <opencv2/core.hpp>

inline float calculateSSD(const cv::Mat& hist1, const cv::Mat& hist2) {
    // Ensure that both histograms have the same size and type
    CV_Assert(hist1.type() == hist2.type() && hist1.size() == hist2.size());

    // Calculate SSD (Sum of Squared Differences)
    cv::Mat diff;
    cv::subtract(hist1, hist2, diff);
    cv::multiply(diff, diff, diff);
    return static_cast<float>(cv::sum(diff)[0]);
}

#endif // CALCULATE_DISTANCE_H