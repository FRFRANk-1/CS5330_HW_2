#include <vector>


/*

  Header file for implementation of a K-means algorithm
*/
#ifndef PROCESS_IMG_FUNS_H
#define PROCESS_IMG_FUNS_H



std :: vector <float> extractBaseLineFeatures(const cv::Mat &image);

std :: vector <std :: pair <std :: string, std :: vector <float>>> computeDataBaseFeatures (const std :: string &directoryPath);

// Function to calculate the sum of squared differences between two feature vectors.
float calculateSSDDistance(const std::vector<float> &feature1, const std::vector<float> &feature2);

// Function to find the closest matches in the database to the target image features.
std::vector<std::pair<std::string, float>> findClosestMatches(
    const std::vector<float> &targetFeatures,
    const std::vector<std::pair<std::string, std::vector<float>>> &databaseFeatures,
    int numberOfOutput);

#endif
