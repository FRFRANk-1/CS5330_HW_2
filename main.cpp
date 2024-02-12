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
#include "multiple_histogram_matching.h"
#include "texture_color.h"
#include "Deep_Embedding.h"
#include "Color_histogram.h"
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
        performBaselineMatchingAndDisplayResults(target_image, image_database_dir, number_of_output, output_file);
        computeAndStoreResultsAndWriteToFile(target_image, image_database_dir, number_of_output, output_file);
        std::cout << "Matching results have been successfully written to [HW_2 -> build -> debug]:" << output_file << std::endl;
         }
        else if (feature_type == "histogram") {
        std :: cout << "histogram:" << std :: endl;
        Histogram_Matching(target_image, image_database_dir, comparisonMethodStr, number_of_output);
        } else if (feature_type == "multiple_histograms") {
        performMultipleHistogramMatching(target_image, image_database_dir, number_of_output);
        } else if (feature_type == "texture_color") {
        texture_color(target_image, image_database_dir, number_of_output);
        } else if (feature_type == "Deep_Embedding") {
            std::string csvFilePath = "D:\\NEU study file\\5330\\HW_2\\ResNet18_olym.csv";
            std::string imageDatabaseDir = argv[2];
            Deep_Embedding(target_image, csvFilePath, imageDatabaseDir, number_of_output);
        } else if (feature_type == "color_histogram" ) {
            //D:\\NEU study file\\5330\\HW_2\\DJI_04233.jpg
        Color_Histogram_Matching(target_image, image_database_dir, comparisonMethodStr, number_of_output);
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
