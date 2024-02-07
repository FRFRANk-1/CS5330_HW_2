#ifndef DIRECTORYREADER_H
#define DIRECTORYREADER_H

#include <windows.h>
#include <string>
#include <vector>

class DirectoryReader {
public:
    DirectoryReader(const std::string& directoryPath);
    ~DirectoryReader();

    // Initializes the directory reading process. Returns true if successful.
    bool openDirectory();

    // Reads the next file in the directory. Returns the file name or an empty string if there are no more files.
    std::string nextFile();

    // Closes the directory.
    void closeDirectory();

private:
    std::string directoryPath;
    HANDLE hFind;
    WIN32_FIND_DATA findData;

    // Helper to construct the search path
    std::string getSearchPath() const;
};

#endif // DIRECTORYREADER_H
