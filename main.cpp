#include "yolov5_dnn.h"
#include <iostream>
#include <fstream>

std::string label_map = "D:/python/yolov5-6.1/uav_bird.txt";
int main(int argc, char** argv) {
	/*std::string names = "10:bike";
	int pos = names.find_first_of(":");
	std::cout << names.substr(0, pos) << " -->> " << names.substr(pos+1) << std::endl;*/
	std::vector<std::string> classNames;
	std::ifstream fp(label_map);
	std::string name;
	while (!fp.eof()) {
		getline(fp, name);
		if (name.length()) {
			classNames.push_back(name);
		}
	}
	fp.close();

	std::shared_ptr<YOLOv5DNNDetector> detector(new YOLOv5DNNDetector());
	detector->initConfig("D:/python/yolov5-6.1/uav_bird_training/uav_bird_best.onnx", 640, 640, 0.25f);
	std::vector<DetectResult> results;
	//cv::Mat frame = cv::imread("D:/test_cu.png");
	//detector->detect(frame, results);
	//for (DetectResult dr : results) {
	//	cv::Rect box = dr.box;
	//	cv::putText(frame, classNames[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
	//}
	//cv::imshow("YOLOv5-6.1 + OpenCV DNN - by gloomyfish", frame);
	//cv::waitKey(0);
	//cv::destroyAllWindows();

	//cv::VideoCapture capture("D:/images/video/sample.mp4");
	cv::VideoCapture capture("D:/bird/uva_test.mp4");
	cv::Mat frame;
	while (true) {
		bool ret = capture.read(frame);
		if (frame.empty()) {
			break;
		}
		detector->detect(frame, results);
		for (DetectResult dr : results) {
			cv::Rect box = dr.box;
			cv::putText(frame, classNames[dr.classId], cv::Point(box.tl().x, box.tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		}
		cv::imshow("YOLOv5-6.1 + OpenCV DNN - by gloomyfish", frame);
		char c = cv::waitKey(50);
		if (c == 27) { // ESC ÍË³ö
			break;
		}
		// reset for next frame
		results.clear();
	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}