Author: Runcheng Li

To use makehist.cpp:
step 1 -> go to CMakeList.txt in add_executable(makeHist.cpp)
step 2 -> build
step 3 -> at terminal, input:".\HW_2.exe "D:\NEU study file\5330\HW_2\Question_9.jpg" "

To use DNG convert to tif:
new window, go to convert_RAW_DNG_py folder
in ternimal -> python raw2tiff.py IMG_1.DNG

To use Kmean.cpp:
step 1 -> go to CMakeList.txt in add_executable(color.cpp kmeans.cpp)
step 2 -> build
step 3 -> at terminal, input:".\HW_2.exe "D:\NEU study file\5330\HW_2\Question_9.jpg" 25"

#open reader
[    // std::string directoryPath = "D:\\NEU study file\\5330\\HW_2\\olympus";
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
]

## task 1 (method 1)
To look up results, which is stored in "D:\\NEU study file\\5330\\HW_2\\build\\Debug"

terminal input:
.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0002.jpg" "D:\NEU study file\5330\HW_2\olympus" "baseline" "SSD" 4  

.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.1016.jpg" "D:\NEU study file\5330\HW_2\olympus" 4 "match_results.txt"

## task 2

terminal input: 
.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0274.jpg" "D:\NEU study file\5330\HW_2\olympus" "histogram" "intersection" 3

.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0164.jpg" "D:\NEU study file\5330\HW_2\olympus" 3 "baseline_result.txt"  

## task 3

terminal input:
.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0274.jpg" "D:\NEU study file\5330\HW_2\olympus" "multiple_histograms" "SSD" "intersection" 4 

$$ task 4

terminal input:
.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0535.jpg" "D:\NEU study file\5330\HW_2\olympus" "texture_color" "SSD" "intersection" 3      

$$ task 5

terminal input:
 .\HW_2.exe "pic.0893.jpg" "D:\NEU study file\5330\HW_2\olympus" "Deep_Embedding" "cosine" "NA" 3   

 .\HW_2.exe "pic.0164.jpg" "D:\NEU study file\5330\HW_2\olympus" "Deep_Embedding" "cosine" "NA" 3 

 ## task 6
terminal input:
 .\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.1072.jpg" "D:\NEU study file\5330\HW_2\olympus" "texture_color" "SSD" "intersection" 3 

 ## extendition:
 