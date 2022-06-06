#include<iostream>
#include "manager.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <map>
#include <cmath>
#include <time.h>
using namespace cv;
using namespace std;

//Note: the input model files can be in *.onnx format;
const char* yolo_engine = "../resources/models/yolov5s.engine";
const char* sort_engine = "../resources/models/deepsort.engine";
float conf_threshold = 0.4;
// output configuration
const char* output_video_name = "result.mp4";
const int fps = 30;

string get_input_video_path(int argc, char *argv[]) {
	string video_source = "";	
	if (argc < 2) {
		std::cout << "[trace] insufficient arguments provided, exiting..." << std::endl;
		return video_source;
	} else {		
		std::cout << "[trace] input offline video source: " << video_source << std::endl;
		return argv[1];
	}
}

int run_main(int argc, char *argv[]) {

	const string video_source = get_input_video_path(argc, argv);
	map<int, vector<int>> personstate;
	map<int, int> classidmap;
	bool is_first = true;
	
	Trtyolosort yosort(yolo_engine, sort_engine);
	VideoCapture capture;
	cv::Mat frame;
	frame = capture.open(video_source);
	if (!capture.isOpened()) {
		std::cerr << "[error] can not open target file: " << video_source << std::endl;
		return -1;
	}
	capture.read(frame);
	const int frame_width = frame.cols;
	const int frame_height = frame.rows;
	std::cout << "[trace] original video dimension: width=" << frame_width << " height=" << frame_height << std::endl;
	
	// configurable part for final output video	
	const int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
	std::vector<DetectBox> det;	
	while (capture.read(frame)) {
		yosort.TrtDetect(frame, conf_threshold, det);		
	}
	capture.release();	
	return 0;
}

int main (int argc, char *argv[]) {

	const int result = run_main(argc, argv);
	return result;
}
