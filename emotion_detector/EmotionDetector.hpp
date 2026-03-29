#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class EmotionDetector {
public:
    explicit EmotionDetector(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "emotion_detector") {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        {
            auto in_name = session_->GetInputNameAllocated(0, allocator);
            input_name_ = in_name.get();
        }
        {
            auto out_name = session_->GetOutputNameAllocated(0, allocator);
            output_name_ = out_name.get();
        }
    }

    std::string predict(const cv::Mat& face_img) {
        if (face_img.empty()) return "No Face";

        cv::Mat resized;
        cv::resize(face_img, resized, cv::Size(kInputWidth, kInputHeight));

        cv::Mat float_img;
        resized.convertTo(float_img, CV_32FC3);

        std::vector<float> input_tensor_values(kInputChannels * kInputHeight * kInputWidth);
        PreprocessToCHW(float_img, input_tensor_values);

        std::array<int64_t, 4> input_shape{1, kInputChannels, kInputHeight, kInputWidth};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size());

        const char* input_names[] = {input_name_.c_str()};
        const char* output_names[] = {output_name_.c_str()};
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1);

        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        size_t class_count = 1;
        for (size_t i = 0; i < output_shape.size(); ++i) {
            if (output_shape[i] > 0) {
                class_count *= static_cast<size_t>(output_shape[i]);
            }
        }
        if (class_count == 0) {
            return "Neutral";
        }

        size_t best_index = 0;
        float best_score = output_data[0];
        for (size_t i = 1; i < class_count; ++i) {
            if (output_data[i] > best_score) {
                best_score = output_data[i];
                best_index = i;
            }
        }

        if (best_score < 0.4f) return "Neutral";

        static const std::vector<std::string> labels = {
            "Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"
        };
        if (best_index >= labels.size()) {
            return "Neutral";
        }
        return labels[best_index];
    }

private:
    static constexpr int kInputChannels = 3;
    static constexpr int kInputHeight = 224;
    static constexpr int kInputWidth = 224;

    void PreprocessToCHW(const cv::Mat& bgr_float_img, std::vector<float>& out) const {
        // Model was trained with RGB channel order and mean subtraction.
        const std::array<float, 3> mean = {123.68f, 116.78f, 103.94f};

        for (int y = 0; y < kInputHeight; ++y) {
            for (int x = 0; x < kInputWidth; ++x) {
                const cv::Vec3f bgr = bgr_float_img.at<cv::Vec3f>(y, x);
                const float r = bgr[2] - mean[0];
                const float g = bgr[1] - mean[1];
                const float b = bgr[0] - mean[2];

                out[0 * kInputHeight * kInputWidth + y * kInputWidth + x] = r;
                out[1 * kInputHeight * kInputWidth + y * kInputWidth + x] = g;
                out[2 * kInputHeight * kInputWidth + y * kInputWidth + x] = b;
            }
        }
    }

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
};