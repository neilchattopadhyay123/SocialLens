#include "attention_demo/attention_estimator.hpp"

#include <algorithm>
#include <cmath>

namespace attention_demo {

void AttentionEstimator::Update(const presage::physiology::MetricsBuffer& metrics, const StatusFlags& flags) {
    const auto now = std::chrono::steady_clock::now();
    const auto blink_window = std::chrono::seconds(static_cast<int>(kBlinkWindowSeconds));

    bool talking_now = false;
    for (const auto& t : metrics.face().talking()) {
        if (t.detected()) {
            talking_now = true;
            break;
        }
    }

    for (const auto& b : metrics.face().blinking()) {
        if (b.detected()) {
            blink_events_.push_back(now);
        }
    }

    while (!blink_events_.empty() && (now - blink_events_.front()) > blink_window) {
        blink_events_.pop_front();
    }

    const double blink_rate_hz = static_cast<double>(blink_events_.size()) / kBlinkWindowSeconds;
    const double blink_score = 1.0 - std::min(1.0, std::abs(blink_rate_hz - 0.25) / 0.25);

    double status_score = 1.0;
    if (flags.face_not_visible) {
        status_score -= 0.60;
    }
    if (flags.too_much_movement) {
        status_score -= 0.25;
    }
    if (flags.poor_lighting) {
        status_score -= 0.20;
    }
    status_score = std::clamp(status_score, 0.0, 1.0);

    const double talking_penalty = talking_now ? 0.35 : 0.0;
    raw_attention_score_ = std::clamp((0.70 * status_score) + (0.30 * blink_score) - talking_penalty, 0.0, 1.0);

    if (!initialized_) {
        attention_score_ = raw_attention_score_;
        initialized_ = true;
        return;
    }

    attention_score_ = kAlpha * raw_attention_score_ + (1.0 - kAlpha) * attention_score_;
}

}  // namespace attention_demo
