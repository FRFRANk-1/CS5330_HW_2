#include "DirectoryReader.h"

bool isImageFile(const std::string& fileName) {
    const std::vector<std::string> imageExtensions = {".jpg", ".png", ".ppm", ".tif"};
    for (const auto& extension : imageExtensions) {
        if (fileName.size() >= extension.size() &&
            fileName.compare(fileName.size() - extension.size(), extension.size(), extension) == 0) {
            return true;
        }
    }
    return false;
}


DirectoryReader::DirectoryReader(const std::string& path)
    : directoryPath(path), hFind(INVALID_HANDLE_VALUE) {
}

DirectoryReader::~DirectoryReader() {
    closeDirectory();
}

bool DirectoryReader::openDirectory() {
    std::string searchPath = getSearchPath();
    hFind = FindFirstFile(searchPath.c_str(), &findData);
    if (hFind == INVALID_HANDLE_VALUE) {
        return false; // Failed to open directory
    }
    return true;
}

std::string DirectoryReader::nextFile() {
    while (FindNextFile(hFind, &findData) != 0) {
        if (!(findData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && isImageFile(findData.cFileName)) {
            return findData.cFileName; // Returns the next file if it's an image.
        }
    }
    return ""; // Returns an empty string if no more files are found or they are not images.
}


void DirectoryReader::closeDirectory() {
    if (hFind != INVALID_HANDLE_VALUE) {
        FindClose(hFind);
        hFind = INVALID_HANDLE_VALUE;
    }
}

std::string DirectoryReader::getSearchPath() const {
    return directoryPath + "\\*"; // Search for all files in the directory
}
