#include <stdlib.h>
#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace cv;
//using namespace cv::ml;

#define PosSamNO 20  //����������
#define NegSamNO 60 //����������

#define CENTRAL_CROP true   //true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����

//HardExample��SVM����������������Щ������Ҫ���ж���ѵ��
//HardExample�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������
//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ
#define HardExampleNO 0  

void set_SVM()
{

	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	CvSVM svm; // SVM������


	string ImgName;//ͼƬ��(����·��)
	ifstream finPos("E:\\��ҵ���\\����\\��ע�ı�\\PositiveImageList.txt");	//������ͼƬ���ļ����б�

	ifstream finNeg("E:\\��ҵ���\\����\\��ע�ı�\\NegativeImageList.txt");	//������ͼƬ���ļ����б�

	Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��	
	Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����


	//���ζ�ȡ������ͼƬ������HOG������
	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		cout<<"����"<<ImgName<<endl;
		ImgName = "E:\\��ҵ���\\����\\�ز�pos֡\\" + ImgName;//������������·����
		Mat src = imread(ImgName);//��ȡͼƬ
		if(CENTRAL_CROP)
			//src = src(Rect(16,16,64,128));//��96*160��INRIA������ͼƬ����Ϊ64*128������ȥ�������Ҹ�16������
			resize(src,src,Size(64,128)); //��������ͼƬ��Ϊ64*128 

		vector<float> descriptors;//HOG����������
		hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		//
		cout<<"������ά����"<<descriptors.size()<<endl;

		//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
		if( 0 == num )
		{
			DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
			//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
			sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
			//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�0��ʾ����
			sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);
		}

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num,i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��
		}
		sampleLabelMat.at<int>(num,0) = 1;//���������Ϊ1������
	}

	//���ζ�ȡ������ͼƬ������HOG������
	for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
	{
		cout<<"����"<<ImgName<<endl;
		ImgName = "E:\\��ҵ���\\����\\�ز�neg֡\\" + ImgName;//���ϸ�������·����
		Mat src = imread(ImgName);//��ȡͼƬ
		resize(src,src,Size(64,128));	//��������ͼƬ��Ϊ64*128 

		vector<float> descriptors;//HOG����������
		hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
		//cout<<"������ά����"<<descriptors.size()<<endl;

		//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];
		}
		//��PosSamNO+num�����������������еĵ�i��Ԫ��
		sampleLabelMat.at<int>(num+PosSamNO,0) = -1;//���������Ϊ-1������
			
	}

	//����HardExample������
	/*if(HardExampleNO > 0)
	{
		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample���������ļ����б�
		//���ζ�ȡHardExample������ͼƬ������HOG������
		for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
		{
			cout<<"����"<<ImgName<<endl;
			ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//����HardExample��������·����
			Mat src = imread(ImgName);//��ȡͼƬ
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			//cout<<"������ά����"<<descriptors.size()<<endl;

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
			{
				sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];
			}
			//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<int>(num+PosSamNO+NegSamNO,0) = -1;//���������Ϊ-1������
		}*/
		

	//���������HOG�������������ļ�
	/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
			fout<<i<<endl;
			for(int j=0; j<DescriptorDim; j++)
			{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
	              		
			}
			fout<<endl;
		}*/

		//ѵ��SVM������
		//SVM��������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01,���η�ʽѡ��϶���ʹ�õĲ���
	CvSVMParams params;	
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
		//svm->setDegree(0);
		//svm->setGamma(1);
		//svm->setCoef0(0);
		//svm->setNu(0);
		//svm->setP(0);
		//svm->setC(0.01);
	params.C = 0.01;

		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

	cout<<"��ʼѵ��SVM������"<<endl;
		//svm->train(sampleFeatureMat, SampleTypes::ROW_SAMPLE, sampleLabelMat);
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), params);

	cout<<"ѵ�����"<<endl;
	svm.save("E:\\��ҵ���\\����\\��ע�ı�\\SVM_HOG_20Pos_60Neg.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�

	system("pause");
}