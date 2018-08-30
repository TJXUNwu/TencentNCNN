#include <stdio.h>
#include <vector>
#include "net.h"
#include "cpu.h"

using namespace std;
using namespace ncnn;
struct Bbox
{
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	bool exist;
	float ppoint[10];
	float regreCoord[4];
};
struct orderScore
{
	float score;
	int oriOrder;
};

class mtcnn
{
public:
	mtcnn();
	void detect(ncnn::Mat& img_, std::vector<Bbox>& finalBbox);	
private:
	void generateBbox(ncnn::Mat score, ncnn::Mat location, vector<Bbox>& boundingBox_, vector<orderScore>& bboxScore_, float scale);
	void nms(vector<Bbox> &boundingBox_, std::vector<orderScore> &bboxScore_, const float overlap_threshold, string modelname = "Union");
	void refineAndSquareBbox(vector<Bbox> &vecBbox, const int &height, const int &width);

	ncnn::Net Pnet, Rnet, Onet;
	ncnn::Mat img;

	float nms_threshold[3];// = { 0.5, 0.7, 0.7 };
	float threshold[3];// = {0.6, 0.6, 0.6};
	float mean_vals[3];// = {127.5, 127.5, 127.5};
	float norm_vals[3];// = {0.0078125, 0.0078125, 0.0078125};
	std::vector<Bbox> firstBbox_, secondBbox_, thirdBbox_;
	std::vector<orderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
	int img_w, img_h;

public:


};









