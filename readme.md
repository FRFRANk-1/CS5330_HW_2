## Author: Runcheng Li
## github_link_this_project: https://github.com/FRFRANk-1/CS5330_HW_2

## Operating system: Window11. IDE: VS code

## CS5330 Project_2: Content-based Image Retrieva

Project due data: 2/10/2024 11:59:59
GradeScope upload date: 2/12/2024
Time travel days Apply for late submission: 2 days

## Instruction to execute Content-based Image Retrieva app
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

## 

## task 1 (method 1)
## baseline_matching.cpp
To look up results, which is stored in "D:\\NEU study file\\5330\\HW_2\\build\\Debug"

terminal input:
.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0164.jpg" "D:\NEU study file\5330\HW_2\olympus" "baseline" "SSD" "CORRELATION" 5."D:/NEU study file/5330/HW_2/build/Debug/HW_2.exe"

.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.1016.jpg" "D:\NEU study file\5330\HW_2\olympus" 4 "match_results.txt"
![Task_1 (IMG-1016 jpg Comparsion)](https://github.com/FRFRANk-1/CS5330_HW_2/assets/116234198/dddef2c6-d728-4167-bc60-85a2656a926d)

## task 2
## Histogram_Matching.cpp

terminal input: 
.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0164.jpg" "D:\NEU study file\5330\HW_2\olympus" "histogram" "SSD" "CHI_SQUARE" 3   

.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0164.jpg" "D:\NEU study file\5330\HW_2\olympus" 3 "baseline_result.txt"  
![Task_2 (IMG-0164 jpg Comparsion)](https://github.com/FRFRANk-1/CS5330_HW_2/assets/116234198/158d6dec-d751-4196-a8fe-35d29cad8bde)

## task 3
## multiple_histogram_matching.cpp

terminal input:
.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0274.jpg" "D:\NEU study file\5330\HW_2\olympus" "multiple_histograms" "SSD" "intersection" 4 
![Task_3 (IMG-0274 jpg Comparsion)](https://github.com/FRFRANk-1/CS5330_HW_2/assets/116234198/e0beeeed-5f21-4aa2-ba5a-dde99fcfa03a)

## task 4
## Color_Histogram.cpp

terminal input:
.\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.0535.jpg" "D:\NEU study file\5330\HW_2\olympus" "texture_color" "SSD" "intersection" 3      
![Task_4 (IMG-0535 jpg Comparsion)](https://github.com/FRFRANk-1/CS5330_HW_2/assets/116234198/d39f15b8-7efa-4ffb-868c-a5f5cca9d1f6)

## task 5
## Deep_Embedding.cpp

terminal input:
 .\HW_2.exe "pic.0893.jpg" "D:\NEU study file\5330\HW_2\olympus" "Deep_Embedding" "cosine" "NA" 3   

 .\HW_2.exe "pic.0164.jpg" "D:\NEU study file\5330\HW_2\olympus" "Deep_Embedding" "cosine" "NA" 3 
![Task_5 (IMG-0164 jpg Comparsion)](https://github.com/FRFRANk-1/CS5330_HW_2/assets/116234198/7bc7e41b-3825-4051-bbea-e12211b56061)
![Task_5 (IMG-0893 jpg Comparsion)](https://github.com/FRFRANk-1/CS5330_HW_2/assets/116234198/47a14225-9a18-4ac1-b833-9cc59bbbe4a8)

## task 6
## comparsion DNN and normal Feature:
![Task_6 (IMG-0734 Comparsion)](https://github.com/FRFRANk-1/CS5330_HW_2/assets/116234198/d3cf50b1-0a34-472d-b004-c67a46b6133f)
![Task_6 (IMG-0948 Comparsion)](https://github.com/FRFRANk-1/CS5330_HW_2/assets/116234198/7e335469-bfe5-4ff7-9c3f-4a1a9c936aa5)
![Uploading Task_6 (IMG-1072 Comparsion).png…]()


## task 7
## texture_color  
terminal input:
 .\HW_2.exe "D:\NEU study file\5330\HW_2\olympus\pic.1072.jpg" "D:\NEU study file\5330\HW_2\olympus" "texture_color" "SSD" "intersection" 3 
![Uploading Task_7 (DJI_04233.jpg Comparsion).png…]()

## extension:
## Banana.cpp

 terminal input:
 .\HW_2.exe "D:\NEU study file\5330\HW_2\banana.jpg" "D:\NEU study file\5330\HW_2\olympus" banana euclidean L2 5

Still debugging~ 
Error: OpenCV(4.9.0) C:\GHA-OCV-1\_work\ci-gha-workflow\ci-gha-workflow\opencv\modules\core\src\matrix_operations.cpp:67: error: (-215:Assertion failed) src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type() in function 'cv::hconcat'
