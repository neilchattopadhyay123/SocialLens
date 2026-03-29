#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class EmotionDetector {
public:
    EmotionDetector(std::string model_path) {
        // Load the ONNX model
        net = cv::dnn::readNetFromONNX(model_path);
        // Optimize for Mac CPU
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    std::string predict(cv::Mat face_img) {
        if (face_img.empty()) return "No Face";

        // AffectNet needs specific preprocessing: 224x224 and color scaling
        cv::Mat blob = cv::dnn::blobFromImage(face_img, 1.0, cv::Size(224, 224), 
                                              cv::Scalar(123.68, 116.78, 103.94), true, false);
        net.setInput(blob);
        cv::Mat prob = net.forward();

        // Get the highest probability
        cv::Point class_id;
        double confidence;
        cv::minMaxLoc(prob, nullptr, &confidence, nullptr, &class_id);

        if (confidence < 0.4) return "Neutral"; // Low confidence defaults to Neutral

        std::vector<std::string> labels = {"Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"};
        return labels[class_id.x];
    }

private:
    cv::dnn::Net net;
};