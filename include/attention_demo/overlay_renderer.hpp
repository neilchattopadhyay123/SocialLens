#pragma once

#include <opencv2/opencv.hpp>

namespace attention_demo {

void DrawAttentionOverlay(cv::Mat& frame, double score, int output_height);

}  // namespace attention_demo
