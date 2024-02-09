#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kmeans.h"
#include "DirectoryReader.h"
#include "process_Img_funs.h"
#include <filesystem>

int main(int argc, char** argv) {

    // std::string directoryPath = "D:\\NEU study file\\5330\\HW_2\\olympus";
    // DirectoryReader reader(directoryPath);

    // if (reader.openDirectory()) {
    //     std::string fileName = reader.nextFile(); // Initial call to get the first file
    //     while (!fileName.empty()) {
    //         std::string fullPath = directoryPath + "\\" + fileName;
    //         std::cout << "Found image file: " << fullPath << std::endl;
    //         fileName = reader.nextFile(); // Get the next file
    //     }
    //     reader.closeDirectory();
    // } else {
    //     std::cerr << "Failed to open directory." << std::endl;
    // }

    // prompt right number of arguments
    if (argc < 6) {
        std :: cerr << " usage:" << argv[0] << "<target_image> <image_database_dir> <feature type> <distance_metric> <number of output>" <<std :: endl;
        return 1;
    }


    // command line arguments
    std :: string target_image = argv[1];
    std :: string image_database_dir = argv[2];
    std :: string feature_type = argv[3];
    std :: string distance_metric = argv[4];
    int number_of_output = std :: stoi(argv[5]); // convert string to int
    
    // debug: print the argumens to verify
    std :: cout << "target image:" << target_image << std :: endl;
    std :: cout << "image database dir:" << image_database_dir << std :: endl;
    std :: cout << "feature type:" << feature_type << std :: endl;
    std :: cout << "distance metric:" << distance_metric << std :: endl;
    std :: cout << "number of output:" << number_of_output << std :: endl;
    
    try {
        // Compute features for the target image
        cv::Mat targetImage = cv::imread(target_image, cv::IMREAD_GRAYSCALE);
        if (targetImage.empty()) {
            throw std::runtime_error("Failed to open target image: " + target_image);
        }
        std::vector<float> targetFeatures = extractBaseLineFeatures(targetImage);

        // Compute features for all images in the database
        std::vector<std::pair<std::string, std::vector<float>>> databaseFeatures = computeDataBaseFeatures(image_database_dir);

        // Find closest matches
        std::vector<MatchResult> matches = computeAndStoreResults(target_image, databaseFeatures, number_of_output);
        // Display matches
        for (const auto& match : matches) {
        std::cout << "Match: " << match.filename << " Distance: " << match.distance << std::endl;
        std::string imagePath = image_database_dir + "/" + match.filename; // Ensure correct path
        cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Could not open or find the image: " << imagePath << std::endl;
            continue; // Skip this iteration if image not found
            }
        cv::imshow("Match: " + match.filename, image);
        cv::waitKey(0); // Wait indefinitely for a key press before moving to the next image
        }

        std::cout << "Writing results to file..." << std::endl;
        std::string outputFile = "match_results.txt";
        computeAndStoreResultsAndWriteToFile(target_image, image_database_dir, number_of_output, outputFile);

            std::filesystem::path cwd = std::filesystem::current_path();

    // Print the current working directory
        std::cout << "Current working directory: " << cwd << std::endl;

        std::cout << "Results written to file: " << outputFile << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }   

    return 0;
}