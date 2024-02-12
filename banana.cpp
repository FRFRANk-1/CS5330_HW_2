#include "banana.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <filesystem>
#include <numeric>
#include <vector>
#include <iostream>

BananaCBIR::BananaCBIR(const std::string& dbDir) : databaseDir(dbDir) {}

void BananaCBIR::buildFeatureDatabase() {
    std::cout << "Building feature database" << std::endl;
    for (const auto& entry : std::filesystem::directory_iterator(databaseDir)) {
        std::string filePath = entry.path().string();
        std::string filename = entry.path().filename().string();

        // Check for image files only (you might want to add more conditions for other image formats)
        if (filename[0] == '.' || (filename.find(".jpg") == std::string::npos && filename.find(".png") == std::string::npos)) {
            std::cout << "Skipping non-image file: " << filename << std::endl;
            continue;
        }

        cv::Mat image = cv::imread(filePath);
        if (image.empty()) {
            std::cerr << "Error: Could not open or find the image: " << filename << std::endl;
            continue;
        }

        // std::cout << "Processing file: " << filename << std::endl;

        cv::Mat colorHist = calculateColorHistogram(image);
        cv::Mat shapeDesc = calculateShapeDescriptor(image);
        cv::Mat textureDesc = calculateTextureDescriptor(image);

        // Ensure the descriptors are of the same type and have a single row for concatenation.
        colorHist.convertTo(colorHist, CV_32F);
        shapeDesc.convertTo(shapeDesc, CV_32F);
        textureDesc.convertTo(textureDesc, CV_32F);
        colorHist = colorHist.reshape(1, 1);
        shapeDesc = shapeDesc.reshape(1, 1);
        textureDesc = textureDesc.reshape(1, 1);

        auto padToMaxCols = [](const cv::Mat &mat, int maxCols) {
    if (mat.cols >= maxCols) return mat;
    int deltaCols = maxCols - mat.cols;
    cv::Mat padded;
    cv::copyMakeBorder(mat, padded, 0, 0, 0, deltaCols, cv::BORDER_CONSTANT, 0);
    return padded;
};
        // std::cout << "maxcols " << filename << std::endl;
        int maxCols = std::max({colorHist.cols, shapeDesc.cols, textureDesc.cols});

        auto padFeatureVector = [maxCols](const cv::Mat& feature) -> cv::Mat {
        if (feature.cols < maxCols) {
        cv::Mat padded;
        cv::copyMakeBorder(feature, padded, 0, 0, 0, maxCols - feature.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
        return padded;
        }
        return feature;
};
        colorHist = padFeatureVector(colorHist);
        shapeDesc = padFeatureVector(shapeDesc);
        textureDesc = padFeatureVector(textureDesc);
        
        // std::cout << "maxcols_1 " << filename << std::endl;

        auto ensureSize = [maxCols](const cv::Mat &feature) {
    if (feature.cols < maxCols) {
        cv::Mat temp(maxCols - feature.cols, 1, feature.type(), cv::Scalar(0));
        return cv::Mat(1, maxCols, feature.type());
        
    }
    return feature;
};

        // Ensure all vectors have the same size
        colorHist = ensureSize(colorHist);
        shapeDesc = ensureSize(shapeDesc);
        textureDesc = ensureSize(textureDesc);

        // Now concatenate the vectors
        std::vector<cv::Mat> features = {colorHist, shapeDesc, textureDesc};
        cv::Mat featureVector;
        cv::hconcat(features.data(), features.size(), featureVector);

        featureDatabase.push_back({filename, featureVector});
    }
    std::cout << "Feature database built successfully." << std::endl;
}

std::vector<BananaMatchResult> BananaCBIR::queryImage(const cv::Mat& queryImage, int topK) {
    cv::Mat queryFeatures = extractFeatures(queryImage);
    std::cout << "debug_test_2." << std::endl;
    std::vector<BananaMatchResult> results;    

    for (const auto& [filename, features] : featureDatabase) {
        float score = compareFeatures(queryFeatures, features);
        results.emplace_back(filename, score);
    }

    std::sort(results.begin(), results.end(), [](const BananaMatchResult& a, const BananaMatchResult& b) {
        return a.distance > b.distance; // higher score comes first
    });

    if (results.size() > static_cast<size_t>(topK)) {
        results.resize(topK);
    }

    return results;
}

cv::Mat BananaCBIR::extractFeatures(const cv::Mat& image) {
    // Combine color, shape, and texture descriptors into a single feature vector
    cv::Mat colorHist = calculateColorHistogram(image);
    cv::Mat shapeDesc = calculateShapeDescriptor(image);
    cv::Mat textureDesc = calculateTextureDescriptor(image);

    cv::Mat features;
    cv::hconcat(std::vector<cv::Mat>{colorHist, shapeDesc, textureDesc}, features);
    return features;
}

cv::Mat BananaCBIR::calculateColorHistogram(const cv::Mat& image) {
    // Convert the image from BGR to HSV color space
    cv::Mat hsvImage;
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HSV);

    // Define the range for yellow color in HSV
    int hMin = 20;  // Hue minimum
    int hMax = 30;  // Hue maximum
    int sMin = 100; // Saturation minimum
    int sMax = 255; // Saturation maximum
    int vMin = 100; // Value minimum
    int vMax = 255; // Value maximum

    // Threshold the HSV image to get only yellow colors
    cv::Mat mask;
    cv::inRange(hsvImage, cv::Scalar(hMin, sMin, vMin), cv::Scalar(hMax, sMax, vMax), mask);

    // Calculate the histogram for the yellow mask
    int hBins = 50; // Number of bins for hue
    int sBins = 60; // Number of bins for saturation
    int histSize[] = {hBins, sBins};

    // Hue varies from 0 to 179, saturation from 0 to 255
    float hRanges[] = {0, 180};
    float sRanges[] = {0, 256};
    const float* ranges[] = {hRanges, sRanges};
    int channels[] = {0, 1}; // Use the 0-th and 1-st channels

    cv::Mat hist;
    cv::calcHist(&hsvImage, 1, channels, mask, hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return hist;
}

cv::Mat BananaCBIR::calculateShapeDescriptor(const cv::Mat& image) {
    // Convert the image to grayscale
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    // Apply a binary threshold to get a binary image
    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // In a simple scenario, assume the largest contour corresponds to the banana
    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2) {
                  return cv::contourArea(c1) > cv::contourArea(c2);
              });

    // Calculate Hu Moments as shape descriptors
    std::vector<double> huMoments(7);
    if (!contours.empty()) {
        cv::Moments moments = cv::moments(contours[0]);
        cv::HuMoments(moments, huMoments);
    }

    // Convert Hu Moments to Mat
    cv::Mat huMomentsMat(huMoments.size(), 1, CV_64F);
    for (size_t i = 0; i < huMoments.size(); i++) {
        huMomentsMat.at<double>(i) = huMoments[i];
    }

    // Log transform Hu Moments to make them scale invariant
    for (int i = 0; i < huMomentsMat.rows; i++) {
        huMomentsMat.at<double>(i) = -1 * copysign(1.0, huMomentsMat.at<double>(i)) * log10(abs(huMomentsMat.at<double>(i)));
    }

    return huMomentsMat;
}

cv::Mat BananaCBIR::calculateTextureDescriptor(const cv::Mat& image) {
    // Convert to grayscale as LBP works on single channel images
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Use Local Binary Patterns for texture description
    cv::Mat lbpImage = gray.clone();
    for (int i = 1; i < gray.rows - 1; i++) {
        for (int j = 1; j < gray.cols - 1; j++) {
            uchar center = gray.at<uchar>(i, j);
            uchar code = 0;
            code |= (gray.at<uchar>(i-1, j-1) > center) << 7;
            code |= (gray.at<uchar>(i-1, j  ) > center) << 6;
            code |= (gray.at<uchar>(i-1, j+1) > center) << 5;
            code |= (gray.at<uchar>(i,   j+1) > center) << 4;
            code |= (gray.at<uchar>(i+1, j+1) > center) << 3;
            code |= (gray.at<uchar>(i+1, j  ) > center) << 2;
            code |= (gray.at<uchar>(i+1, j-1) > center) << 1;
            code |= (gray.at<uchar>(i,   j-1) > center) << 0;
            
            lbpImage.at<uchar>(i-1,j-1) = code;
        }
    }

    // Calculate the histogram of the LBP image
    int histSize = 256; // number of bins
    float range[] = {0, 256}; // the upper boundary is exclusive
    const float* histRange = {range};
    cv::Mat lbpHist;
    cv::calcHist(&lbpImage, 1, 0, cv::Mat(), lbpHist, 1, &histSize, &histRange, true, false);

    // Normalize the histogram so that images of different sizes can be compared
    cv::normalize(lbpHist, lbpHist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    return lbpHist;
}

float BananaCBIR::compareFeatures(const cv::Mat& features1, const cv::Mat& features2) {
    // Ensure the feature vectors are of the same size and type
    if (features1.rows != features2.rows || features1.cols != features2.cols || features1.type() != features2.type()) {
        throw std::invalid_argument("Feature vectors have different sizes or types and cannot be compared.");
    }

    // Normalize the feature vectors
    cv::Mat normalizedFeatures1, normalizedFeatures2;
    cv::normalize(features1, normalizedFeatures1);
    cv::normalize(features2, normalizedFeatures2);

    double dotProduct = normalizedFeatures1.dot(normalizedFeatures2);

    float cosineSimilarity = static_cast<float>(dotProduct);

    float similarityMeasure = (1.0f + cosineSimilarity) / 2.0f;

    return similarityMeasure;
}

void banana_matching(const std::string& target_image_path, const std::string& image_database_dir, int top_k) {
    // Load the target image
    cv::Mat target_image = cv::imread(target_image_path);
    if (target_image.empty()) {
        throw std::runtime_error("Could not open or find the target image!");
    }

    // Create BananaCBIR instance
    BananaCBIR cbir(image_database_dir);

    // Build the feature database
    cbir.buildFeatureDatabase();

    // Query the target image
    std::vector<BananaMatchResult> results = cbir.queryImage(target_image, top_k);

    // Display the top_k results
    for (const auto& result : results) {
        std::cout << "Match: " << result.filename << " with distance: " << result.distance << std::endl;
        cv::Mat img = cv::imread(image_database_dir + "/" + result.filename);
        cv::imshow("Match - " + result.filename, img);
    }
    cv::waitKey(0); // Wait for a key press to close the windows
    cv::destroyAllWindows(); // Close all the opened windows

}
