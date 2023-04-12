#pragma once

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

struct DetectResult {
	int classId;
	float score;
	cv::Rect box;
};

class YOLOv5OpenVINODetector {
public:
	void initConfig(std::string onnxpath, float confidence_threshold, float score_threshold);
	void detect(cv::Mat & frame, std::vector<DetectResult> &results);
private:
	float confidence_threshold = 0.4;
	float score_threshold = 0.25;
	std::string input_name = "images";
	std::string out_name = "output";
	ov::InferRequest infer_request;
};