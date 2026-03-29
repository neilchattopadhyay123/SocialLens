#pragma once

#include "attention_demo/status_flags.hpp"

#include <physiology/modules/messages/metrics.h>

#include <chrono>
#include <deque>

namespace attention_demo {

class AttentionEstimator {
public:
    void Update(const presage::physiology::MetricsBuffer& metrics, const StatusFlags& flags);

    double attention_score() const { return attention_score_; }
    double raw_attention_score() const { return raw_attention_score_; }

private:
    std::deque<std::chrono::steady_clock::time_point> blink_events_;
    double attention_score_ = 0.0;
    double raw_attention_score_ = 0.0;
    bool initialized_ = false;

    static constexpr double kAlpha = 0.2;
    static constexpr double kBlinkWindowSeconds = 6.0;
};

}  // namespace attention_demo
