#include "attention_demo/overlay_renderer.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace attention_demo {

void DrawAttentionOverlay(cv::Mat& frame, double score, int output_height) {
    const int percent = static_cast<int>(std::round(score * 100.0));
    const std::string label = "Attention: " + std::to_string(percent) + "%";

    const cv::Point bar_origin(30, output_height - 50);
    const int bar_width = 360;
    const int bar_height = 18;
    const int fill_width = static_cast<int>(bar_width * score);

    cv::rectangle(frame, cv::Rect(bar_origin.x, bar_origin.y, bar_width, bar_height), cv::Scalar(80, 80, 80), cv::FILLED);
    cv::Scalar fill_color(0, 180, 0);
    if (score < 0.6) {
        fill_color = cv::Scalar(0, 200, 255);
    }
    if (score < 0.35) {
        fill_color = cv::Scalar(0, 80, 255);
    }

    cv::rectangle(frame, cv::Rect(bar_origin.x, bar_origin.y, std::max(0, fill_width), bar_height), fill_color, cv::FILLED);

    cv::putText(frame, label, cv::Point(30, output_height - 62), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
}

}  // namespace attention_demo
