#include <iostream>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include "kmeans.h"
#include "DirectoryReader.h"

int main(int argc, char** argv) {

    std::string directoryPath = "D:\\NEU study file\\5330\\HW_2\\olympus";
    DirectoryReader reader(directoryPath);

    if (reader.openDirectory()) {
        std::string fileName = reader.nextFile(); // Initial call to get the first file
        while (!fileName.empty()) {
            std::string fullPath = directoryPath + "\\" + fileName;
            std::cout << "Found image file: " << fullPath << std::endl;
            fileName = reader.nextFile(); // Get the next file
        }
        reader.closeDirectory();
    } else {
        std::cerr << "Failed to open directory." << std::endl;
    }

    return 0;
}