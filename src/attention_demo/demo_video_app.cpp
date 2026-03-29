#include "attention_demo/demo_video_app.hpp"

#include "attention_demo/attentiveness_scorer.hpp"
#include "attention_demo/overlay_renderer.hpp"
#include "EmotionDetector.hpp"

#include <smartspectra/container/foreground_container.hpp>
#include <smartspectra/container/settings.hpp>
#include <physiology/modules/messages/metrics.h>
#include <physiology/modules/messages/status.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace attention_demo {

using namespace presage::smartspectra;

int DemoVideoApp::Run(const std::string& api_key,
                     const std::string& input_video_path,
                     const std::string& output_video_path) {
    try {
        // Probe input FPS so output timing matches source video.
        double output_fps = 30.0;
        {
            cv::VideoCapture probe(input_video_path);
            if (probe.isOpened()) {
                const double probed_fps = probe.get(cv::CAP_PROP_FPS);
                if (probed_fps >= 1.0 && probed_fps <= 240.0) {
                    output_fps = probed_fps;
                }
            }
        }

        container::settings::Settings<
            container::settings::OperationMode::Continuous,
            container::settings::IntegrationMode::Rest
        > settings;

        settings.video_source.device_index = -1;
        settings.video_source.input_video_path = input_video_path;
        settings.video_source.input_video_time_path = "";
        settings.video_source.capture_width_px = 1280;
        settings.video_source.capture_height_px = 720;
        settings.video_source.auto_lock = false;

        settings.headless = false;
        settings.enable_edge_metrics = true;
        settings.verbosity_level = 1;
        settings.interframe_delay_ms = static_cast<int>(std::round(1000.0 / output_fps));
        settings.continuous.preprocessed_data_buffer_duration_s = 0.5;
        settings.integration.api_key = api_key;

        auto container = std::make_unique<container::CpuContinuousRestForegroundContainer>(settings);

        const int output_width = 1280;
        const int output_height = 720;
        const cv::Size output_size(output_width, output_height);

        cv::VideoWriter writer;
        bool opened = false;
        int selected_fourcc = 0;
        for (const int candidate_fourcc : {
                 cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                 cv::VideoWriter::fourcc('m', 'p', '4', 'v')
             }) {
            if (writer.open(output_video_path, candidate_fourcc, output_fps, output_size, true)) {
                selected_fourcc = candidate_fourcc;
                opened = true;
                break;
            }
        }

        if (!opened) {
            std::cerr << "Failed to open output video for writing: " << output_video_path << "\n";
            return 1;
        }
        std::cout << "Video writer configured at " << output_fps << " FPS (codec fourcc=" << selected_fourcc << ")\n";

        std::string emotion_model_path;
        for (const std::string& candidate : {
                 std::string("emotion_detector/emo_affectnet_opencv.onnx"),
                 std::string("../emotion_detector/emo_affectnet_opencv.onnx"),
                 std::string("emotion_detector/emo_affectnet.onnx"),
                 std::string("../emotion_detector/emo_affectnet.onnx"),
                 std::string("./emo_affectnet.onnx")
             }) {
            if (std::filesystem::exists(candidate)) {
                emotion_model_path = candidate;
                break;
            }
        }
        std::unique_ptr<EmotionDetector> emotion_detector;
        bool emotion_enabled = false;
        if (emotion_model_path.empty()) {
            std::cout << "Warning: emotion model file not found; continuing without emotion detection.\n";
        } else {
            try {
                emotion_detector = std::make_unique<EmotionDetector>(emotion_model_path);
                emotion_enabled = true;
            } catch (const std::exception& e) {
                std::cout << "Warning: failed to initialize ONNX emotion detector (" << e.what()
                          << "); continuing without emotion detection.\n";
            }
        }

        cv::CascadeClassifier face_cascade;
        bool face_cascade_loaded = false;
        for (const std::string& candidate : {
                 std::string("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
                 std::string("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
                 std::string("haarcascade_frontalface_default.xml")
             }) {
            if (std::filesystem::exists(candidate) && face_cascade.load(candidate)) {
                face_cascade_loaded = true;
                break;
            }
        }
        if (!face_cascade_loaded) {
            std::cout << "Warning: Haar cascade not found; emotion classification will use full frame.\n";
        }

        AttentivenessScorer scorer;
        std::deque<std::string> emotion_history;
        const size_t emotion_history_size = 60;

        auto status = container->SetOnCoreMetricsOutput(
            [&scorer](const presage::physiology::MetricsBuffer& metrics, int64_t) {
                scorer.OnCoreMetrics(metrics);
                return absl::OkStatus();
            }
        );
        if (!status.ok()) {
            std::cerr << "Failed to set metrics callback: " << status.message() << "\n";
            return 1;
        }

        status = container->SetOnEdgeMetricsOutput(
            [&scorer](const presage::physiology::Metrics& metrics, int64_t) {
                scorer.OnEdgeMetrics(metrics);
                return absl::OkStatus();
            }
        );
        if (!status.ok()) {
            std::cerr << "Failed to set edge metrics callback: " << status.message() << "\n";
            return 1;
        }

        status = container->SetOnStatusChange(
            [&scorer](presage::physiology::StatusValue imaging_status) {
                scorer.OnStatusChange(imaging_status);
                return absl::OkStatus();
            }
        );
        if (!status.ok()) {
            std::cerr << "Failed to set status callback: " << status.message() << "\n";
            return 1;
        }

        status = container->SetOnVideoOutput(
            [&writer,
             &scorer,
             &emotion_history,
             &emotion_detector,
             emotion_enabled,
             emotion_history_size,
             &face_cascade,
             face_cascade_loaded,
             output_width,
             output_height](cv::Mat& frame, int64_t) {
                if (frame.cols != output_width || frame.rows != output_height) {
                    cv::resize(frame, frame, cv::Size(output_width, output_height));
                }

                const auto attentiveness = scorer.Compute();
                const double score = static_cast<double>(attentiveness.score / 100.f);

                std::string emotion = emotion_enabled ? "Unknown" : "Unavailable";
                if (emotion_enabled && emotion_detector && !frame.empty()) {
                    cv::Mat face_crop = frame;
                    if (face_cascade_loaded) {
                        cv::Mat gray;
                        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                        std::vector<cv::Rect> faces;
                        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(80, 80));

                        if (!faces.empty()) {
                            int best_idx = 0;
                            int best_area = std::numeric_limits<int>::min();
                            for (size_t i = 0; i < faces.size(); ++i) {
                                const int area = faces[i].width * faces[i].height;
                                if (area > best_area) {
                                    best_area = area;
                                    best_idx = static_cast<int>(i);
                                }
                            }
                            const cv::Rect clamped = faces[best_idx] & cv::Rect(0, 0, frame.cols, frame.rows);
                            face_crop = frame(clamped);
                        }
                    }

                    try {
                        emotion = emotion_detector->predict(face_crop);
                    } catch (const std::exception&) {
                        emotion = "Unavailable";
                    }
                }

                // Smooth label jitter by displaying the most frequent emotion over
                // the last N frames instead of the instantaneous prediction.
                emotion_history.push_back(emotion);
                if (emotion_history.size() > emotion_history_size) {
                    emotion_history.pop_front();
                }

                std::unordered_map<std::string, int> counts;
                for (const auto& e : emotion_history) {
                    counts[e]++;
                }

                std::string smoothed_emotion = emotion;
                int best_count = -1;
                for (auto it = emotion_history.rbegin(); it != emotion_history.rend(); ++it) {
                    const int c = counts[*it];
                    if (c > best_count) {
                        best_count = c;
                        smoothed_emotion = *it;
                    }
                }

                cv::putText(frame,
                            "Emotion: " + smoothed_emotion,
                            cv::Point(30, 40),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.9,
                            cv::Scalar(255, 255, 255),
                            2,
                            cv::LINE_AA);

                DrawAttentionOverlay(frame, score, output_height);
                writer.write(frame);
                return absl::OkStatus();
            }
        );
        if (!status.ok()) {
            std::cerr << "Failed to set video callback: " << status.message() << "\n";
            return 1;
        }

        std::cout << "Initializing demo video generator...\n";
        if (auto init_status = container->Initialize(); !init_status.ok()) {
            std::cerr << "Failed to initialize: " << init_status.message() << "\n";
            return 1;
        }

        std::cout << "Generating demo video from: " << input_video_path << "\n";
        if (auto run_status = container->Run(); !run_status.ok()) {
            std::cerr << "Processing failed: " << run_status.message() << "\n";
            return 1;
        }

        writer.release();
        std::cout << "Demo video created: " << output_video_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}

}  // namespace attention_demo
