#ifndef PROCESS_IMG_FUNS_H
#define PROCESS_IMG_FUNS_H

#include <vector>
#include <string>
#include <opencv2/core.hpp>

struct MatchResult {
    std::string filename;
    float distance;
    MatchResult(std::string f = "", float d = 0.0f) : filename(f), distance(d) {}
    bool operator<(const MatchResult& other) const { return distance < other.distance; }
};

std::vector<float> extractBaseLineFeatures(const cv::Mat &image);
std::vector<std::pair<std::string, std::vector<float>>> computeDataBaseFeatures(const std::string &directoryPath);
std::vector<MatchResult> computeAndStoreResults(const std::string& target_image_path, const std::vector<std::pair<std::string, std::vector<float>>>& databaseFeatures, int number_of_output);
void computeAndStoreResultsAndWriteToFile(const std::string& target_image, const std::string& image_database_dir, int number_of_output, const std::string& output_file);
float calculateSSDDistance(const std::vector<float> &feature1, const std::vector<float> &feature2);
std::vector<std::pair<std::string, float>> findClosestMatches(const std::vector<float> &targetFeatures, const std::vector<std::pair<std::string, std::vector<float>>> &databaseFeatures, int numberOfOutput);

// Histogram functions
cv::Mat computeRGBHistogram(const cv::Mat& image, int bins = 8);
std::vector<std::pair<std::string, cv::Mat>> computeDataBaseHistograms(const std::string &directoryPath, int bins = 8);
std::vector<MatchResult> findClosestHistogramMatches(const cv::Mat& targetHistogram, const std::vector<std::pair<std::string, cv::Mat>>& databaseHistograms, int number_of_output);
double histogramIntersection(const cv::Mat& hist1, const cv::Mat& hist2);

#endif // PROCESS_IMG_FUNS_H
