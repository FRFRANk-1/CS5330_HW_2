#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kmeans.h"
#include "DirectoryReader.h"
#include "baseline_matching.h"
#include "histogram_matching.h"
#include <filesystem>

int main(int argc, char** argv) {

    if (argc < 7) {
        std :: cerr << " usage:" << argv[0] << "<target_image> <image_database_dir> <feature type> <distance_metric> <comparisonMethodStr> <number of output>" <<std :: endl;
        return 1;
    }

    // command line arguments
    std :: string target_image = argv[1];
    std :: string image_database_dir = argv[2];
    std :: string feature_type = argv[3];
    std :: string distance_metric = argv[4];
    std :: string comparisonMethodStr = argv[5];
    int number_of_output = std :: stoi(argv[6]); // convert string to int
    
    // debug: print the argumens to verify
    std :: cout << "target image:" << target_image << std :: endl;
    std :: cout << "image database dir:" << image_database_dir << std :: endl;
    std :: cout << "feature type:" << feature_type << std :: endl;
    std :: cout << "distance metric:" << distance_metric << std :: endl;
    std :: cout << "comparison method:" << comparisonMethodStr << std :: endl;
    std :: cout << "number of output:" << number_of_output << std :: endl;
    std :: string output_file = "output.txt";

    std::vector<MatchResult> matches;

    try {
        if (feature_type == "baseline") {
        std::vector<std::pair<std::string, std::vector<float>>> databaseFeatures = computeDataBaseFeatures(image_database_dir);
        matches = computeAndStoreResults(target_image, databaseFeatures, number_of_output);
        printFeatureComparisons(target_image, databaseFeatures);
        
        std::cout << "Top " << number_of_output << " closest matches to the target image:" << std::endl;
        for (const auto& match : matches) {
            std::cout << "Image: " << match.filename << ", Distance: " << match.distance << std::endl;
        }

        displayTopMatches(matches, image_database_dir, number_of_output);
        
        computeAndStoreResultsAndWriteToFile(target_image, image_database_dir, number_of_output, output_file);
        std::cout << "Matching results have been successfully written to [HW_2 -> build -> debug]:" << output_file << std::endl;
         }
        else if (feature_type == "histogram") {
        std :: cout << "histogram:" << std :: endl;
        Histogram_Matching(target_image, image_database_dir, comparisonMethodStr, number_of_output);
        }
        else {
            std :: cerr << "Invalid feature type: " << feature_type << std :: endl;
            return -1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }   

    return 0;
}
