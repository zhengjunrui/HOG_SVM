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
	CvCapture *capture = cvCreateFileCapture("E:\\��ҵ���\\����\\������Ƶ\\test.avi");
	//CvCapture *capture = cvCreateFileCapture("E:\\������\\��Ŀ\\ͼ��Ԥ����\\��һ����Ƶ����\\ˤ��.avi");
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

		resize(image1,image,cvSize(500,300),400,400,CV_INTER_LINEAR);		//̫��Ļ��ܲ���
		// 1. ����HOG����  
		HOGDescriptor hog; // ����Ĭ�ϲ���  
  
		// 2. ����SVM������  
		hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());   // ��ʱ����OpenCVĬ�ϵ����˼�������  
  
		// 3. �ڲ���ͼ���ϼ����������  
		vector<Rect> regions;  
		hog.detectMultiScale(image, regions, 0, Size(8, 8), Size(32, 32), 1.05, 1);  
		
		//Mat red_rect = cvCreateMat(image1.rows,image1.cols,;

		//��ʾ���Σ�Ĭ�ϵľ��ο�̫���ֶ�ʹ���С 

		for(int i=0; i<regions.size(); i++)		
		{
			Rect r = regions[i];
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(image, r.tl(), r.br(), Scalar(0,0,255), 2);		//tl()�������Ͻ����꣬br()�������½�����
		}

		/*for (size_t i = 0; i < regions.size(); i++)  
		{  
		    rectangle(image, regions[i], Scalar(0, 0, 255), 2);  //���ж������˵�����һ�����ν��б�� 
		} */ 

		//IplImage *frame_new;
		*frame = IplImage(image);
		//cvWriteFrame(writer,frame);		//����Ƶ֡����writer
		cvShowImage("PeopleDetection", frame);  
		//waitKey(0);  
		//system("pause");
		//cvShowImage("����",frame);
		char c = cvWaitKey(10);		//ʵ�ʲ�����Ҫ33��һ��30֡��
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