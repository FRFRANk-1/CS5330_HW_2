#ifndef BANANA_H
#define BANANA_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct BananaMatchResult {
    std::string filename;
    float distance;

    BananaMatchResult() : filename(""), distance(0.0f) {} // Default constructor
    BananaMatchResult(std::string fn, float dist) : filename(std::move(fn)), distance(dist) {}
};


class BananaCBIR {
public:
    explicit BananaCBIR(const std::string& databaseDir);
    void buildFeatureDatabase();
    std::vector<BananaMatchResult> queryImage(const cv::Mat& queryImage, int topK = 5);
    static cv::Mat extractFeatures(const cv::Mat& image);

private:
    std::string databaseDir;
    std::vector<std::pair<std::string, cv::Mat>> featureDatabase;

    static cv::Mat calculateColorHistogram(const cv::Mat& image);
    static cv::Mat calculateShapeDescriptor(const cv::Mat& image);
    static cv::Mat calculateTextureDescriptor(const cv::Mat& image);
    static float compareFeatures(const cv::Mat& features1, const cv::Mat& features2);
    void padFeatureVector(cv::Mat& feature, int maxCols);
};

void banana_matching(const std::string& target_image_path, const std::string& image_database_dir, int top_k);

#endif // BANANA_H