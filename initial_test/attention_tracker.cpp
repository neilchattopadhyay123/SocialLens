// attention_tracker.cpp
// SmartSpectra attention tracker using a test video source.

#include <smartspectra/container/foreground_container.hpp>
#include <smartspectra/container/settings.hpp>
#include <smartspectra/gui/opencv_hud.hpp>
#include <physiology/modules/messages/metrics.h>
#include <physiology/modules/messages/status.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <deque>
#include <iostream>
#include <mutex>
#include <string>

using namespace presage::smartspectra;

class EWMA {
public:
    explicit EWMA(double alpha) : alpha_(alpha) {}

    double Update(double value) {
        if (!initialized_) {
            value_ = value;
            initialized_ = true;
            return value_;
        }
        value_ = alpha_ * value + (1.0 - alpha_) * value_;
        return value_;
    }

private:
    double alpha_ = 0.2;
    double value_ = 0.0;
    bool initialized_ = false;
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    std::string api_key;
    if (argc > 1) {
        api_key = argv[1];
    } else if (const char* env_key = std::getenv("SMARTSPECTRA_API_KEY")) {
        api_key = env_key;
    } else {
        std::cout << "Usage: ./attention_tracker YOUR_API_KEY\n";
        std::cout << "Or set SMARTSPECTRA_API_KEY environment variable\n";
        std::cout << "Get your API key from: https://physiology.presagetech.com\n";
        return 1;
    }

    try {
        container::settings::Settings<
            container::settings::OperationMode::Continuous,
            container::settings::IntegrationMode::Rest
        > settings;

        settings.video_source.device_index = -1;
        settings.video_source.input_video_path = "test.mp4";
        settings.video_source.input_video_time_path = "";
        settings.video_source.capture_width_px = 1280;
        settings.video_source.capture_height_px = 720;
        settings.video_source.auto_lock = false;

        settings.headless = false;
        settings.enable_edge_metrics = true;
        settings.verbosity_level = 1;
        settings.continuous.preprocessed_data_buffer_duration_s = 0.5;
        settings.integration.api_key = api_key;

        auto container = std::make_unique<container::CpuContinuousRestForegroundContainer>(settings);
        auto hud = std::make_unique<gui::OpenCvHud>(10, 0, 1260, 400);

        const int window_width = 1280;
        const int window_height = 720;
        const std::string window_name = "SmartSpectra Attention Tracker";
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(window_name, window_width, window_height);

        std::mutex attention_mutex;
        double attention_score = 0.0;
        double raw_attention_score = 0.0;
        bool face_not_visible = false;
        bool too_much_movement = false;
        bool poor_lighting = false;

        EWMA smoother(0.2);
        std::deque<std::chrono::steady_clock::time_point> blink_events;

        auto status = container->SetOnCoreMetricsOutput(
            [&hud,
             &attention_mutex,
             &attention_score,
             &raw_attention_score,
             &smoother,
             &blink_events,
             &face_not_visible,
             &too_much_movement,
             &poor_lighting](const presage::physiology::MetricsBuffer& metrics, int64_t) {
                hud->UpdateWithNewMetrics(metrics);

                const auto now = std::chrono::steady_clock::now();
                const auto blink_window = std::chrono::seconds(6);

                bool talking_now = false;
                for (const auto& t : metrics.face().talking()) {
                    if (t.detected()) {
                        talking_now = true;
                        break;
                    }
                }

                for (const auto& b : metrics.face().blinking()) {
                    if (b.detected()) {
                        blink_events.push_back(now);
                    }
                }

                while (!blink_events.empty() && (now - blink_events.front()) > blink_window) {
                    blink_events.pop_front();
                }

                const double blink_rate_hz = static_cast<double>(blink_events.size()) / 6.0;
                double blink_score = 1.0 - std::min(1.0, std::abs(blink_rate_hz - 0.25) / 0.25);

                double status_score = 1.0;
                if (face_not_visible) {
                    status_score -= 0.60;
                }
                if (too_much_movement) {
                    status_score -= 0.25;
                }
                if (poor_lighting) {
                    status_score -= 0.20;
                }
                status_score = std::clamp(status_score, 0.0, 1.0);

                const double talking_penalty = talking_now ? 0.35 : 0.0;
                const double raw = std::clamp((0.70 * status_score) + (0.30 * blink_score) - talking_penalty, 0.0, 1.0);
                const double smooth = smoother.Update(raw);

                {
                    std::lock_guard<std::mutex> lock(attention_mutex);
                    raw_attention_score = raw;
                    attention_score = smooth;
                }

                std::cout << "Attention: " << static_cast<int>(smooth * 100.0) << "%\n";
                return absl::OkStatus();
            }
        );
        if (!status.ok()) {
            std::cerr << "Failed to set metrics callback: " << status.message() << "\n";
            return 1;
        }

        status = container->SetOnStatusChange(
            [&attention_mutex, &face_not_visible, &too_much_movement, &poor_lighting](presage::physiology::StatusValue imaging_status) {
                const auto code = imaging_status.value();
                std::string status_description = presage::physiology::GetStatusDescription(code);
                std::string status_lower = status_description;
                std::transform(status_lower.begin(), status_lower.end(), status_lower.begin(),
                               [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

                const bool has_face_not_visible =
                    status_lower.find("face") != std::string::npos &&
                    status_lower.find("visible") != std::string::npos;
                const bool has_too_much_movement =
                    status_lower.find("movement") != std::string::npos ||
                    status_lower.find("motion") != std::string::npos;
                const bool has_poor_lighting =
                    status_lower.find("light") != std::string::npos;

                {
                    std::lock_guard<std::mutex> lock(attention_mutex);
                    face_not_visible = has_face_not_visible;
                    too_much_movement = has_too_much_movement;
                    poor_lighting = has_poor_lighting;
                }

                std::cout << "Status: " << status_description << "\n";
                return absl::OkStatus();
            }
        );
        if (!status.ok()) {
            std::cerr << "Failed to set status callback: " << status.message() << "\n";
            return 1;
        }

        status = container->SetOnVideoOutput(
            [&hud,
             &attention_mutex,
             &attention_score,
             &raw_attention_score,
             window_width,
             window_height,
             window_name](cv::Mat& frame, int64_t) {
                if (frame.cols != window_width || frame.rows != window_height) {
                    cv::resize(frame, frame, cv::Size(window_width, window_height));
                }

                if (auto render_status = hud->Render(frame); !render_status.ok()) {
                    std::cerr << "HUD render failed: " << render_status.message() << "\n";
                }

                double score = 0.0;
                double raw = 0.0;
                {
                    std::lock_guard<std::mutex> lock(attention_mutex);
                    score = attention_score;
                    raw = raw_attention_score;
                }

                const int percent = static_cast<int>(std::round(score * 100.0));
                const std::string label = "Attention: " + std::to_string(percent) + "%";

                const cv::Point bar_origin(30, window_height - 50);
                const int bar_width = 360;
                const int bar_height = 18;
                const int fill_width = static_cast<int>(bar_width * score);

                cv::rectangle(
                    frame,
                    cv::Rect(bar_origin.x, bar_origin.y, bar_width, bar_height),
                    cv::Scalar(80, 80, 80),
                    cv::FILLED
                );
                cv::Scalar fill_color(0, 180, 0);
                if (score < 0.6) {
                    fill_color = cv::Scalar(0, 200, 255);
                }
                if (score < 0.35) {
                    fill_color = cv::Scalar(0, 80, 255);
                }
                cv::rectangle(
                    frame,
                    cv::Rect(bar_origin.x, bar_origin.y, std::max(0, fill_width), bar_height),
                    fill_color,
                    cv::FILLED
                );
                cv::rectangle(
                    frame,
                    cv::Rect(bar_origin.x, bar_origin.y, bar_width, bar_height),
                    cv::Scalar(255, 255, 255),
                    1
                );

                cv::putText(
                    frame,
                    label,
                    cv::Point(30, window_height - 62),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 255, 255),
                    2,
                    cv::LINE_AA
                );

                cv::putText(
                    frame,
                    "Raw: " + std::to_string(static_cast<int>(std::round(raw * 100.0))) + "%",
                    cv::Point(30, window_height - 20),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    cv::Scalar(220, 220, 220),
                    1,
                    cv::LINE_AA
                );

                cv::imshow(window_name, frame);
                const char key = static_cast<char>(cv::waitKey(1) & 0xFF);
                if (key == 'q' || key == 27) {
                    return absl::CancelledError("User quit");
                }
                return absl::OkStatus();
            }
        );
        if (!status.ok()) {
            std::cerr << "Failed to set video callback: " << status.message() << "\n";
            return 1;
        }

        std::cout << "Initializing attention tracker...\n";
        if (auto init_status = container->Initialize(); !init_status.ok()) {
            std::cerr << "Failed to initialize: " << init_status.message() << "\n";
            return 1;
        }

        std::cout << "Running attention tracker. Press 'q' to quit.\n";
        if (auto run_status = container->Run(); !run_status.ok()) {
            std::cerr << "Processing failed: " << run_status.message() << "\n";
            return 1;
        }

        cv::destroyAllWindows();
        std::cout << "Done!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
