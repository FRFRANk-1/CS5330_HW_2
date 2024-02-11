#include "histogram_matching.h"
#include <filesystem>

HistogramMatchResult::HistogramMatchResult(std::string f, float s) : filename(f), similarity(s) {}

cv::Mat calcHistForImage(const cv::Mat& image, int bins) {
    cv::Mat hist;
    int histSize[] = {bins};
    float range[] = {0, 256};
    const float* ranges[] = {range};
    int channels[] = {0};

    // Compute histogram
    cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

    return hist;
}

float compareHistograms(const cv::Mat& hist1, const cv::Mat& hist2, HistogramComparisonMethod method) {
    return cv::compareHist(hist1, hist2, static_cast<int>(method));
}

void Histogram_Matching(const std::string& target_histogram_image, const std::string& image_database_dir, const std::string& comparisonMethodStr, int number_of_output) {
    // Convert comparisonMethodStr to HistogramComparisonMethod enum
    HistogramComparisonMethod comparisonMethod;
    if (comparisonMethodStr == "CORRELATION") {
        comparisonMethod = CORRELATION;
    } else if (comparisonMethodStr == "CHI_SQUARE") {
        comparisonMethod = CHI_SQUARE;
    } else if (comparisonMethodStr == "INTERSECTION") {
        comparisonMethod = INTERSECTION;
    } else if (comparisonMethodStr == "BHATTACHARYYA") {
        comparisonMethod = BHATTACHARYYA;
    } else {
        std::cerr << "Invalid histogram comparison method: " << comparisonMethodStr << std::endl;
        return;
    }

    // Load the target image
    cv::Mat target_image = cv::imread(target_histogram_image, cv::IMREAD_COLOR);
    if (target_image.empty()) {
        std::cerr << "Failed to load target image: " << target_histogram_image << std::endl;
        return;
    }
    cv::Mat target_hist = calcHistForImage(target_image, 16); // 32 bins

    std::vector<HistogramMatchResult> matches;

    // Process images in the database
    for (const auto& entry : std::filesystem::directory_iterator(image_database_dir)) {
        std::string filepath = entry.path().string();
        if (filepath != target_histogram_image) {
            cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Failed to load image: " << filepath << std::endl;
                continue;
            }
            cv::Mat hist = calcHistForImage(image, 16); //  32 bins
            float similarity = compareHistograms(target_hist, hist, comparisonMethod);
            matches.emplace_back(filepath, similarity);
        }
    }

    // Sort matches based on similarity
    std::sort(matches.begin(), matches.end(), [](const HistogramMatchResult& a, const HistogramMatchResult& b) {
        return a.similarity > b.similarity;
    });

    // Display results
    for (int i = 0; i < std::min(number_of_output, static_cast<int>(matches.size())); ++i) {
        std::cout << "Match: " << matches[i].filename << ", Similarity: " << matches[i].similarity << std::endl;
    }
    // Display the target image
    cv::namedWindow("Target Image", cv::WINDOW_AUTOSIZE);
    cv::imshow("Target Image", target_image);

    // Display the top N matches
    for (int i = 0; i < std::min(number_of_output, static_cast<int>(matches.size())); i++) {
        std::string matchFilePath = matches[i].filename;
        cv::Mat matchImage = cv::imread(matchFilePath, cv::IMREAD_COLOR);
        if (!matchImage.empty()) {
            std::string windowName = "Match " + std::to_string(i + 1);
            cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
            cv::imshow(windowName, matchImage);
            std::cout << windowName << ": " << matchFilePath << ", Similarity: " << matches[i].similarity << std::endl;
        } else {
            std::cerr << "Failed to load image: " << matchFilePath << std::endl;
        }
    }

    cv::waitKey(0); // Wait for a keystroke in the window
    cv::destroyAllWindows(); // Close all OpenCV windows
}
