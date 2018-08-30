#include <stdio.h>
#include <vector>
#include <iostream>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mtcnn.h"
#pragma comment( lib, "vfw32.lib" ) 
#pragma comment( lib, "comctl32.lib" ) 
//#define USE_CAMERA
using namespace std;
using namespace cv;
using namespace ncnn;


int main(int argc, char** argv)
{	
#ifdef USE_CAMERA //ʹ������ͷ
	cv::namedWindow("Win7x64", WINDOW_NORMAL);
	//setWindowProperty("frame", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);����//���ô���ȫ��
	cv::VideoCapture capture;
	cv::Mat camera;
	//���� Directshow �ķ�ʽ�򿪵�һ������ͷ�豸��
	capture.open(0);
	if (!capture.isOpened())
	{
		return -1;
	}
	bool bret = capture.set(CV_CAP_PROP_FRAME_WIDTH, 1600);
	bret = capture.set(CV_CAP_PROP_FRAME_HEIGHT, 1200);
	//capture.set(CAP_PROP_SETTINGS,0);//���� Directshow ����ͷ����������
	mtcnn Net;
	while (true)
	{
		//��ȡһ֡ͼ��
		capture.read(camera);
		if (camera.empty())
		{
			continue;
		}		
		std::printf("--- img w: %d  h:%d  ch:%d\n", camera.cols, camera.rows, camera.channels());
		float start = clock();
		int times = 1;
		ncnn::set_omp_num_threads(4);
		for (int cnt = 0; cnt < times; cnt++)
		{
			std::vector<Bbox> finalBbox;
			//OpenCV������ͼƬ��BGR��ʽ�ģ���ҪתΪRGB��ʽ���������ʻ�ܵ͡�  
			ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(camera.data, ncnn::Mat::PIXEL_BGR2RGB, camera.cols, camera.rows);
			Net.detect(ncnn_img, finalBbox);
			for (vector<Bbox>::iterator it = finalBbox.begin(); it != finalBbox.end(); it++)
			{
				if ((*it).exist)
				{
					//std::printf("Bbox [x1,y1], [x2,y2]:[%d,%d], [%d,%d] \n", (*it).x1, (*it).x2, (*it).y1, (*it).y2);
					rectangle(camera, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0, 0, 255), 2, 8, 0);
					for (int num = 0; num < 5; num++)
					{
						//std::printf("Landmark [x1,y1]: [%d,%d] \n", (int)*(it->ppoint + num), (int)*(it->ppoint + num + 5));
						circle(camera, Point((int)*(it->ppoint + num), (int)*(it->ppoint + num + 5)), 3, Scalar(0, 255, 255), -1);
					}
				}
			}
			std::vector<Bbox>().swap(finalBbox);
			ncnn_img.release();
		}
		std::printf("MTCNN mean time comsuming: %f ms\n", (clock() - start) / times);
		cv::imshow("Win7x64", camera);		
		if (cv::waitKey(1) == 27)
		{
			break;
		}
	}
	
#else //����ͼƬ�ļ�
	const char* imagepath;// = argv[1];
	if (argc == 2)
	{
		imagepath = argv[1];
	}
	else
	{
		imagepath = "1.jpg";
	}
	cout << imagepath << endl;
	cv::Mat cv_img = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
	if (cv_img.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		system("pause");
		return -1;
	}
	std::printf("img w: %d  h:%d  ch:%d\n", cv_img.cols, cv_img.rows, cv_img.channels());
	//cv::imshow("img", cv_img);
	//cv::waitKey(10);
	float start = clock();
	int times = 1;
	ncnn::set_omp_num_threads(4);
	for (int cnt = 0; cnt < times; cnt++)
	{
		std::vector<Bbox> finalBbox;	
		mtcnn Net;
		//OpenCV������ͼƬ��BGR��ʽ�ģ���ҪתΪRGB��ʽ���������ʻ�ܵ͡�  
		ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(cv_img.data, ncnn::Mat::PIXEL_BGR2RGB, cv_img.cols, cv_img.rows);
		Net.detect(ncnn_img, finalBbox);
		for (vector<Bbox>::iterator it = finalBbox.begin(); it != finalBbox.end(); it++)
		{
			if ((*it).exist)
			{
				std::printf("Bbox [x1,y1], [x2,y2]:[%d,%d], [%d,%d] \n", (*it).x1, (*it).x2, (*it).y1, (*it).y2);
				rectangle(cv_img, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0, 0, 255), 2, 8, 0);
				for (int num = 0; num < 5; num++)
				{
					std::printf("Landmark [x1,y1]: [%d,%d] \n", (int)*(it->ppoint + num), (int)*(it->ppoint + num + 5));
					circle(cv_img, Point((int)*(it->ppoint + num), (int)*(it->ppoint + num + 5)), 3, Scalar(0, 255, 255), -1);
				}
			}
		}
	}
	std::printf("MTCNN mean time comsuming: %f ms\n", (clock() - start) / times);
	cv::imshow("result.jpg", cv_img);
	cv::waitKey(0);
#endif
	system("pause");
	return 0;
}






