#include "highgui.h"
#include "cv.h"
#include "cvaux.h" 
#include <ctype.h>  
#include <stdlib.h>
#include <iostream>
using namespace cv;
using namespace std;

void hog_svm()
{
	cvNamedWindow("PeopleDetection",CV_WINDOW_NORMAL);
	CvCapture *capture = cvCreateFileCapture("E:\\毕业设计\\代码\\测试视频\\test.avi");
	//CvCapture *capture = cvCreateFileCapture("E:\\大三上\\项目\\图像预处理\\第一次视频拍摄\\摔倒.avi");
	//CvVideoWriter* writer = cvCreateVideoWriter("test_detector.avi",CV_FOURCC('M', 'J', 'P', 'G'),25,cvSize(500,300),1);
	IplImage *frame;
	//IplImage *frame_new=0;
	
	while(1)
	{
		frame = cvQueryFrame(capture);
		if(!frame)
			break;
		Mat image1 = (Mat)(frame);
		Mat image;

		resize(image1,image,cvSize(500,300),400,400,CV_INTER_LINEAR);		//太大的话跑不动
		// 1. 定义HOG对象  
		HOGDescriptor hog; // 采用默认参数  
  
		// 2. 设置SVM分类器  
		hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());   // 暂时采用OpenCV默认的行人检测分类器  
  
		// 3. 在测试图像上检测行人区域  
		vector<Rect> regions;  
		hog.detectMultiScale(image, regions, 0, Size(8, 8), Size(32, 32), 1.05, 1);  
		
		//Mat red_rect = cvCreateMat(image1.rows,image1.cols,;

		//显示矩形，默认的矩形框太大，手动使其变小 

		for(int i=0; i<regions.size(); i++)		
		{
			Rect r = regions[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(image, r.tl(), r.br(), Scalar(0,0,255), 2);		//tl()返回左上角坐标，br()返回右下角坐标
		}

		/*for (size_t i = 0; i < regions.size(); i++)  
		{  
		    rectangle(image, regions[i], Scalar(0, 0, 255), 2);  //对判定是行人的区域画一个矩形进行标记 
		} */ 

		//IplImage *frame_new;
		*frame = IplImage(image);
		//cvWriteFrame(writer,frame);		//将视频帧存入writer
		cvShowImage("PeopleDetection", frame);  
		//waitKey(0);  
		//system("pause");
		//cvShowImage("测试",frame);
		char c = cvWaitKey(10);		//实际参数需要33（一秒30帧）
		if(c==27)
		{
			break;
		}

	}
	cvReleaseCapture(&capture);
	cvDestroyWindow("PeopleDetection");
	cvWaitKey(0);
	system("pause");
	//return frame;
}