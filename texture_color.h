#ifndef TEXTURE_COLOR_H
#define TEXTURE_COLOR_H

#include <opencv2/core.hpp>
#include <vector>

struct TextureColorMatchResult {
    std::string filename;
    float similarity;
    
    TextureColorMatchResult(std::string f, float s);
};

    // Function to compute the whole image color histogram
    cv::Mat CalColorHistogram(const cv::Mat& image);

    // Function to compute the whole image texture histogram using Sobel gradients
    cv::Mat CalTextureHistogram(const cv::Mat& image);

    // Function to compute the combined feature vector
    std::vector<float> computeCombinedFeatureVector(const cv::Mat& colorHist, const cv::Mat& textureHist, float colorWeight = 0.5f);

    void texture_color(const std::string& target_texture_color_image, const std::string& image_database_dir, int number_of_output);

#endif // TEXTURE_COLOR_H

    // std::cout << "Image type: " << image.type() << std::endl;
    // std::cout << "Image channels: " << image.channels() << std::endl;
    // std::cout << "Image size: " << image.size() << std::endl;