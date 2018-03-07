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
  
    // 1. ����HOG����  
    HOGDescriptor hog; // ����Ĭ�ϲ���  
  
    // 2. ����SVM������  
    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());   // ��ʱ����OpenCVĬ�ϵ����˼�������  
  
     // 3. �ڲ���ͼ���ϼ����������  
    vector<Rect> regions;  
    hog.detectMultiScale(image, regions, 0, Size(8, 8), Size(32, 32), 1.05, 1);  
  
    // ��ʾ  
    for (size_t i = 0; i < regions.size(); i++)  
    {  
        rectangle(image, regions[i], Scalar(0, 0, 255), 2);  //���ж������˵�����һ�����ν��б�� 
    }  
  
    imshow("���˼��", image);  
    waitKey(0);  
	system("pause");
}