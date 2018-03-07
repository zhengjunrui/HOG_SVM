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

#define PosSamNO 20  //正样本个数
#define NegSamNO 60 //负样本个数

#define CENTRAL_CROP true   //true:训练时，对96*160的INRIA正样本图片剪裁出中间的64*128大小人体

//HardExample是SVM分类错误的样本，这些样本需要进行二次训练
//HardExample：负样本个数。如果HardExampleNO大于0，表示处理完初始负样本集后，继续处理HardExample负样本集。
//不使用HardExample时必须设置为0，因为特征向量矩阵和特征类别矩阵的维数初始化时用到这个值
#define HardExampleNO 0  

void set_SVM()
{

	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	CvSVM svm; // SVM分类器


	string ImgName;//图片名(绝对路径)
	ifstream finPos("E:\\毕业设计\\代码\\标注文本\\PositiveImageList.txt");	//正样本图片的文件名列表

	ifstream finNeg("E:\\毕业设计\\代码\\标注文本\\NegativeImageList.txt");	//负样本图片的文件名列表

	Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数	
	Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人


	//依次读取正样本图片，生成HOG描述子
	for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
	{
		cout<<"处理："<<ImgName<<endl;
		ImgName = "E:\\毕业设计\\代码\\素材pos帧\\" + ImgName;//加上正样本的路径名
		Mat src = imread(ImgName);//读取图片
		if(CENTRAL_CROP)
			//src = src(Rect(16,16,64,128));//将96*160的INRIA正样本图片剪裁为64*128，即剪去上下左右各16个像素
			resize(src,src,Size(64,128)); //将正样本图片缩为64*128 

		vector<float> descriptors;//HOG描述子向量
		hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
		//
		cout<<"描述子维数："<<descriptors.size()<<endl;

		//处理第一个样本时初始化特征向量矩阵和类别矩阵，因为只有知道了特征向量的维数才能初始化特征向量矩阵
		if( 0 == num )
		{
			DescriptorDim = descriptors.size();//HOG描述子的维数
			//初始化所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数sampleFeatureMat
			sampleFeatureMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, DescriptorDim, CV_32FC1);
			//初始化训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，0表示无人
			sampleLabelMat = Mat::zeros(PosSamNO + NegSamNO + HardExampleNO, 1, CV_32SC1);
		}

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num,i) = descriptors[i];//第num个样本的特征向量中的第i个元素
		}
		sampleLabelMat.at<int>(num,0) = 1;//正样本类别为1，有人
	}

	//依次读取负样本图片，生成HOG描述子
	for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
	{
		cout<<"处理："<<ImgName<<endl;
		ImgName = "E:\\毕业设计\\代码\\素材neg帧\\" + ImgName;//加上负样本的路径名
		Mat src = imread(ImgName);//读取图片
		resize(src,src,Size(64,128));	//将正样本图片缩为64*128 

		vector<float> descriptors;//HOG描述子向量
		hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
		//cout<<"描述子维数："<<descriptors.size()<<endl;

		//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
		for(int i=0; i<DescriptorDim; i++)
		{
			sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];
		}
		//第PosSamNO+num个样本的特征向量中的第i个元素
		sampleLabelMat.at<int>(num+PosSamNO,0) = -1;//负样本类别为-1，无人
			
	}

	//处理HardExample负样本
	/*if(HardExampleNO > 0)
	{
		ifstream finHardExample("HardExample_2400PosINRIA_12000NegList.txt");//HardExample负样本的文件名列表
		//依次读取HardExample负样本图片，生成HOG描述子
		for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
		{
			cout<<"处理："<<ImgName<<endl;
			ImgName = "D:\\DataSet\\HardExample_2400PosINRIA_12000Neg\\" + ImgName;//加上HardExample负样本的路径名
			Mat src = imread(ImgName);//读取图片
			//resize(src,img,Size(64,128));

			vector<float> descriptors;//HOG描述子向量
			hog.compute(src,descriptors,Size(8,8));//计算HOG描述子，检测窗口移动步长(8,8)
			//cout<<"描述子维数："<<descriptors.size()<<endl;

			//将计算好的HOG描述子复制到样本特征矩阵sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
			{
				sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];
			}
			//第PosSamNO+num个样本的特征向量中的第i个元素
			sampleLabelMat.at<int>(num+PosSamNO+NegSamNO,0) = -1;//负样本类别为-1，无人
		}*/
		

	//输出样本的HOG特征向量矩阵到文件
	/*	ofstream fout("SampleFeatureMat.txt");
		for(int i=0; i<PosSamNO+NegSamNO; i++)
		{
			fout<<i<<endl;
			for(int j=0; j<DescriptorDim; j++)
			{	fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
	              		
			}
			fout<<endl;
		}*/

		//训练SVM分类器
		//SVM参数：设SVM类型为C_SVC；线性核函数；松弛因子C=0.01,调参方式选择较多人使用的参数
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

		//迭代终止条件，当迭代满1000次或误差小于FLT_EPSILON时停止迭代
		
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, FLT_EPSILON);

	cout<<"开始训练SVM分类器"<<endl;
		//svm->train(sampleFeatureMat, SampleTypes::ROW_SAMPLE, sampleLabelMat);
	svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), params);

	cout<<"训练完成"<<endl;
	svm.save("E:\\毕业设计\\代码\\标注文本\\SVM_HOG_20Pos_60Neg.xml");//将训练好的SVM模型保存为xml文件

	system("pause");
}