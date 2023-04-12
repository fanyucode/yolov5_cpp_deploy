#include <yolov5_openvino.h>

void YOLOv5OpenVINODetector::initConfig(std::string onnxpath, float conf, float scored) {
	this->confidence_threshold = conf;
	this->score_threshold = scored;
	ov::Core ie;
	std::vector<std::string> devices = ie.get_available_devices();
	for (std::string name : devices) {
		std::cout << "device name: " << name << std::endl;
	}

	ov::CompiledModel compiled_model = ie.compile_model(onnxpath, "CPU");
	this->infer_request = compiled_model.create_infer_request();
}

void YOLOv5OpenVINODetector::detect(cv::Mat &frame, std::vector<DetectResult> &results) {
	// 获取输入格式
	int64 start = cv::getTickCount();
	// 请求网络输入
	ov::Tensor input_tensor = infer_request.get_input_tensor();
	ov::Shape tensor_shape = input_tensor.get_shape();
	size_t num_channels = tensor_shape[1];
	size_t input_h = tensor_shape[2];
	size_t input_w = tensor_shape[3];

	// 图象预处理 - 格式化操作
	int w = frame.cols;
	int h = frame.rows;
	int _max = std::max(h, w);
	cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
	cv::Rect roi(0, 0, w, h);
	frame.copyTo(image(roi));

	float x_factor = image.cols / static_cast<float>(input_w);
	float y_factor = image.rows / static_cast<float>(input_h);

	size_t image_size = input_w * input_h;
	cv::Mat blob_image;
	resize(image, blob_image, cv::Size(input_w, input_h));
	blob_image.convertTo(blob_image, CV_32F);
	blob_image = blob_image / 255.0;

	// NCHW 设置输入图象数据
	float* data = input_tensor.data<float>();
	for (size_t row = 0; row < input_h; row++) {
		for (size_t col = 0; col < input_w; col++) {
			for (size_t ch = 0; ch < num_channels; ch++) {
				data[image_size*ch + row * input_w + col] = blob_image.at<cv::Vec3f>(row, col)[ch];
			}
		}
	}

	// 推理与返回结果
	this->infer_request.infer();

	auto output = this->infer_request.get_tensor("output");
	const float* prob = (float*)output.data();
	const ov::Shape outputDims = output.get_shape();
	size_t numRows = outputDims[1];
	size_t numCols = outputDims[2];

	// 后处理, 1x25200x85
	std::vector<cv::Rect> boxes;
	std::vector<int> classIds;
	std::vector<float> confidences;
	cv::Mat det_output(numRows, numCols, CV_32F, (float*)prob);
	for (int i = 0; i < det_output.rows; i++) {
		float confidence = det_output.at<float>(i, 4);
		if (confidence < this->confidence_threshold) {
			continue;
		}
		cv::Mat classes_scores = det_output.row(i).colRange(5, numCols);
		cv::Point classIdPoint;
		double score;
		minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

		// 置信度 0～1之间
		if (score > this->score_threshold)
		{
			float cx = det_output.at<float>(i, 0);
			float cy = det_output.at<float>(i, 1);
			float ow = det_output.at<float>(i, 2);
			float oh = det_output.at<float>(i, 3);
			int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
			int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
			int width = static_cast<int>(ow * x_factor);
			int height = static_cast<int>(oh * y_factor);
			// printf("cx:%.2f, cy:%.2f, ow:%.2f, oh:%.2f, x_factor:%.2f, y_factor:%.2f \n", cx, cy, ow, oh, x_factor, y_factor);
			cv::Rect box;
			box.x = x;
			box.y = y;
			box.width = width;
			box.height = height;

			boxes.push_back(box);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
		}
	}

	// NMS
	std::vector<int> indexes;
	cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		DetectResult dr;
		int index = indexes[i];
		int idx = classIds[index];
		dr.box = boxes[index];
		dr.classId = idx;
		dr.score = confidences[index];
		cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
		cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
			cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
		results.push_back(dr);
	}
	// 计算FPS render it
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
}