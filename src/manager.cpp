#include "manager.hpp"
#include <assert.h>     /* assert */

using std::vector;
using namespace cv;
static Logger gLogger;


const char* coco_classes[] = { "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",      "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",   "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush" };

const cv::Scalar palette[] = {
	cv::Scalar(189, 68, 252),
	cv::Scalar(169, 41, 33),
	cv::Scalar(2, 98, 98),
	cv::Scalar(135, 151, 81),
	cv::Scalar(155, 215, 229),
	cv::Scalar(206, 44, 69),
	cv::Scalar(44, 52, 105),
	cv::Scalar(134, 1, 178),
	cv::Scalar(172, 243, 129),
	cv::Scalar(44, 117, 205),
	cv::Scalar(233, 166, 155),
	cv::Scalar(71, 27, 83),
	cv::Scalar(168, 179, 184),
	cv::Scalar(118, 9, 180),
	cv::Scalar(191, 148, 203),
	cv::Scalar(220, 201, 103),
	cv::Scalar(224, 24, 131),
	cv::Scalar(143, 118, 5),
	cv::Scalar(23, 53, 170),
	cv::Scalar(73, 17, 138),
	cv::Scalar(237, 205, 121),
	cv::Scalar(177, 133, 175),
	cv::Scalar(51, 7, 60),
	cv::Scalar(10, 233, 91),
	cv::Scalar(172, 77, 225),
	cv::Scalar(67, 247, 177),
	cv::Scalar(12, 242, 184),
	cv::Scalar(105, 244, 106),
	cv::Scalar(102, 214, 61),
	cv::Scalar(38, 68, 36),
	cv::Scalar(166, 117, 144),
	cv::Scalar(60, 167, 159),
	cv::Scalar(224, 102, 131),
	cv::Scalar(176, 70, 160),
	cv::Scalar(153, 237, 188),
	cv::Scalar(69, 141, 157),
	cv::Scalar(150, 225, 38),
	cv::Scalar(74, 14, 127),
	cv::Scalar(121, 8, 195),
	cv::Scalar(121, 136, 123),
	cv::Scalar(51, 114, 66),
	cv::Scalar(140, 2, 4),
	cv::Scalar(210, 48, 86),
	cv::Scalar(129, 58, 106),
	cv::Scalar(206, 2, 214),
	cv::Scalar(152, 7, 89),
	cv::Scalar(91, 217, 57),
	cv::Scalar(47, 117, 244),
	cv::Scalar(219, 190, 37),
	cv::Scalar(219, 210, 226),
	cv::Scalar(116, 37, 102),
	cv::Scalar(58, 184, 94),
	cv::Scalar(108, 2, 243),
	cv::Scalar(148, 95, 212),
	cv::Scalar(181, 81, 142),
	cv::Scalar(149, 60, 182),
	cv::Scalar(159, 59, 56),
	cv::Scalar(18, 124, 210),
	cv::Scalar(172, 193, 41),
	cv::Scalar(218, 150, 64),
	cv::Scalar(230, 11, 137),
	cv::Scalar(21, 54, 32),
	cv::Scalar(28, 81, 99),
	cv::Scalar(248, 208, 72),
	cv::Scalar(131, 171, 32),
	cv::Scalar(106, 148, 203),
	cv::Scalar(95, 152, 20),
	cv::Scalar(245, 55, 88),
	cv::Scalar(132, 150, 220),
	cv::Scalar(140, 91, 142),
	cv::Scalar(182, 18, 233),
	cv::Scalar(85, 232, 107),
	cv::Scalar(128, 177, 240),
	cv::Scalar(150, 222, 190),
	cv::Scalar(68, 245, 33),
	cv::Scalar(36, 148, 116),
	cv::Scalar(18, 205, 130),
	cv::Scalar(213, 144, 22),
	cv::Scalar(64, 229, 229),
	cv::Scalar(103, 69, 75),
};

// Display UI configuration
#define IOU_BORDER_THICKNESS 1.5
#define LABEL_FONT_THICKNESS 2

Trtyolosort::Trtyolosort(const char* yolo_engine_path, const char* sort_engine_path) {
	
	this->sort_engine_path_ = sort_engine_path;
	this->yolo_engine_path_ = yolo_engine_path;
	trt_engine = yolov5_trt_create(yolo_engine_path_);	
	std::cout << "[trace] creating DeepSort instance using " << sort_engine_path_ << std::endl;
	DS = new DeepSort(sort_engine_path_, 128, 256, 0, &gLogger);
}

const char* get_label_by_id(const int index)
{
	const int array_len = sizeof(coco_classes) / sizeof(char*);
	if (0 <= index && index < array_len) {
		const char* name_cstr = coco_classes[index];
		return name_cstr;
	}
	return "unknown";
}

const cv::Scalar get_color_by_id(const int index) {

	const int array_len = sizeof(palette) / sizeof(cv::Scalar);
	if (0 <= index && index < array_len) {
		return palette[index];
	}
	return cv::Scalar(0, 0, 0);
}

std::map<int, int> class_counter;
void Trtyolosort::showDetection(cv::Mat& img, std::vector<DetectBox>& boxes, const int fps) { 
  
	cv::Mat temp = img.clone();
	for (auto box : boxes) {
		cv::Point lt(box.x1, box.y1);
		cv::Point br(box.x2, box.y2);
		const int class_id = (int)box.classID;
		const int track_id = (int)box.trackID;

		const std::string label = cv::format("%s", get_label_by_id(class_id));
		const cv::Scalar _color = get_color_by_id(class_id);
		cv::rectangle(temp, lt, br, _color, IOU_BORDER_THICKNESS);
		cv::putText(temp, label, lt, cv::FONT_HERSHEY_SIMPLEX, 1, _color, LABEL_FONT_THICKNESS);
		class_counter[class_id] ++;
	}

	std::string jetson_fps = "JetsonTX2 FPS: " + std::to_string(fps);
	cv::putText(temp, jetson_fps, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2.5, cv::LINE_4);
	std::string total_objs = "Object count: " + std::to_string(boxes.size());
	cv::putText(temp, total_objs, cv::Point(10, 100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2.5, cv::LINE_4);

	int current_pos = 150;
	int height = 30;
	for (auto it = class_counter.begin(); it != class_counter.end(); it++) {
		const int key= it->first;
		const int value = it->second;		
		const char* name = get_label_by_id(key);
		const std::string counting = cv::format("%s: %d", name, value);
		cv::putText(temp, counting, cv::Point(10, current_pos), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2.5, cv::LINE_4);
		current_pos += height;
	}

	class_counter.clear();
	//write the output to the console or disk file;
	if (this->_video_writer != nullptr) {
		this->_video_writer->write(temp);
	} else {
		cv::imshow("Objects tracking result", temp);	
	}
	cv::waitKey(1);	
}

int Trtyolosort::TrtDetect(cv::Mat& frame, float& conf_thresh, std::vector<DetectBox>& det) {
	
	auto start = std::chrono::system_clock::now();
	// yolo detect
	auto ret = yolov5_trt_detect(trt_engine, frame, conf_thresh, det);
	// deepsort detect 
	DS->sort(frame, det);    
	auto end = std::chrono::system_clock::now();
	const int delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();	
	const int fps = 1000.0 / (delta);
	showDetection(frame, det, fps);
	return 1;

}

void Trtyolosort::setVideoWriter(cv::VideoWriter* video_writer) {
	this->_video_writer = video_writer;	
}


