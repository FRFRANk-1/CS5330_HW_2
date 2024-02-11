#include "multiple_histogram_matching.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>

    cv::Mat calculateHistogram(const cv::Mat& image, int bins=8) {
    cv::Mat hist;
    int histSize[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range, range, range};
    int channels[] = {0, 1, 2};

    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    return hist;
    }

    cv::Mat createHistImage(const cv::Mat& hist) {
    cv::Mat norm_hist;
    cv::normalize(hist, norm_hist, 0, 255, cv::NORM_MINMAX);

    int histSize = hist.rows;
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound((double) hist_w / histSize);

    // Create an image to visualize the histogram
    cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage, cv::Point(bin_w * (i - 1), hist_h - cvRound(norm_hist.at<float>(i - 1))),
             cv::Point(bin_w * (i), hist_h - cvRound(norm_hist.at<float>(i))),
             cv::Scalar(255), 2, 8, 0);
        }
        return histImage;
    }

// Main function to perform matching
void performMultipleHistogramMatching(const std::string& target_image_path, const std::string& image_database_dir, int number_of_output) {
    // Load the target image
    cv::Mat target_image = cv::imread(target_image_path);
    if (target_image.empty()) {
        throw std::runtime_error("Failed to load target image.");
    }
    
    std::vector<multipleHistogramMatch> matches;

    std::sort(matches.begin(), matches.end(), [](const multipleHistogramMatch& a, const multipleHistogramMatch& b) {
        return a.distance < b.distance;
    });

    std::cout << "Comparing features with target image: " << target_image_path << std::endl;

    // Convert target image to different color spaces and compute histograms
    cv::Mat target_rgb, target_hsv;
    cv::cvtColor(target_image, target_rgb, cv::COLOR_BGR2RGB);
    cv::cvtColor(target_image, target_hsv, cv::COLOR_BGR2HSV);
    
    std::cout << "RGB: " << target_rgb.size() << ", HSV: " << target_hsv.size() << std::endl;

    // cv::Mat mask; // Define a mask if focusing on specific image parts
    cv::Mat hist_rgb = calculateHistogram(target_rgb, cv::COLOR_RGB2BGR);
    cv::Mat hist_hsv = calculateHistogram(target_hsv, cv::COLOR_HSV2BGR);
    cv::Mat hist_rgb_center = calculateHistogram(target_rgb(cv::Rect(target_rgb.cols/2 - 50, target_rgb.rows/2 - 50, 100, 100)), cv::COLOR_RGB2BGR); // Center image histogram

    std::cout << "RGB hist: " << hist_rgb.size() << ", HSV hist: " << hist_hsv.size() << std::endl;

    for (int i = 0; i < std::min(number_of_output, static_cast<int>(matches.size())); ++i) {
        const auto& match = matches[i];
        std::string matchFilePath = matches[i].filename;
        cv::Mat matchImage = cv::imread(matchFilePath, cv::IMREAD_COLOR);
        
        if (!matchImage.empty()) {
            cv::Mat hist = calculateHistogram(target_image, 8); 
            cv::Mat histImage = createHistImage(hist); // Create visual representation of the histogram

            std::string windowName = "Match " + std::to_string(i + 1) + ": " + match.filename;
            cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
            cv::imshow(windowName, histImage);
        } else {
            std::cerr << "Failed to load image at " << matchFilePath << std::endl;
        }
    }

    // Display histograms
    cv::namedWindow("RGB Center Image Histogram");
    cv::imshow("RGB Center Image Histogram", hist_rgb_center);
    cv::waitKey(0);
    // Iterate through the image database and compute similarity
    

    // Calculate distances
    double distance_rgb = cv::norm(hist_rgb, hist_rgb_center, cv::NORM_L1);
    double distance_hsv = cv::norm(hist_hsv, hist_hsv, cv::NORM_L1);

    // Output distances
    std::cout << "Distance (RGB): " << distance_rgb << std::endl;
    std::cout << "Distance (HSV): " << distance_hsv << std::endl;
    
    // Placeholder for similarity computation and result storage
}

cv::Mat calculateHistogram(const cv::Mat& image, int colorSpace, const cv::Mat& mask, int bins) {
    cv::Mat hist;
    // Histogram calculation logic
    return hist;
}
