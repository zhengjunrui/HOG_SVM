#include "cv.h"
#include "highgui.h"
#include <ctype.h>  
#include <stdlib.h>  
#include "cvaux.h" 
using namespace cv;
using namespace std;

void hogsvm()
{
	Mat image = imread("E:\\video\\frame\\1.jpg");  
  
    if (image.empty())  
    {  
        cout << "read imagefailed" << endl;  
    }  
  
    // 1. 定义HOG对象  
    HOGDescriptor hog; // 采用默认参数  
  
    // 2. 设置SVM分类器  
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());   // 暂时采用OpenCV默认的行人检测分类器  
  
     // 3. 在测试图像上检测行人区域  
    vector<Rect> regions;  
    hog.detectMultiScale(image, regions, 0, Size(8, 8), Size(32, 32), 1.05, 1);  
  
    // 显示  
    for (size_t i = 0; i < regions.size(); i++)  
    {  
        rectangle(image, regions[i], Scalar(0, 0, 255), 2);  //对判定是行人的区域画一个矩形进行标记 
    }  
  
    imshow("行人检测", image);  
    waitKey(0);  
	system("pause");
}