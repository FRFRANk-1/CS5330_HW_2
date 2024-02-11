#ifndef MULTIPLE_HISTOGRAM_MATCHING_H
#define MULTIPLE_HISTOGRAM_MATCHING_H

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct multipleHistogramMatch {
    std::string filename;
    float distance;

    multipleHistogramMatch(std::string f, float d);
};


// Function declarations
void performMultipleHistogramMatching(const std::string& target_image_path, const std::string& image_database_dir, int number_of_output);

#endif // MULTIPLE_HISTOGRAM_MATCHING_H
