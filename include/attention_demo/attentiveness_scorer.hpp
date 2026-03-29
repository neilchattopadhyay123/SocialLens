#pragma once

#include <physiology/modules/messages/metrics.h>
#include <physiology/modules/messages/status.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <deque>
#include <mutex>
#include <optional>
#include <numeric>
#include <cstddef>
#include <string>
#include <vector>

namespace attention_demo {

struct SubScores {
    float presence_score = 0.f;
    float breathing_score = 0.f;
    float blink_score = 0.f;

    float face_present_frac = 0.f;
    float breathing_rate_bpm = 0.f;
    float blink_rate_bpm = 0.f;
    size_t blink_event_count = 0;

    bool breathing_valid = false;
    bool blink_valid = false;
    bool presence_valid = false;
};

struct AttentivenessResult {
    float score = 0.f;
    SubScores sub;
    bool valid = false;
    std::chrono::steady_clock::time_point timestamp;
};

struct ScorerConfig {
    std::chrono::seconds window{30};

    float w_presence = 0.40f;
    float w_breathing = 0.35f;
    float w_blink = 0.25f;

    float presence_full = 0.80f;

    float breath_drowsy_lo = 8.f;
    float breath_alert_lo = 12.f;
    float breath_alert_hi = 22.f;
    float breath_stressed = 28.f;

    float blink_very_low = 3.f;
    float blink_focus_lo = 6.f;
    float blink_focus_hi = 15.f;
    float blink_normal_hi = 25.f;
    float blink_fatigue = 35.f;

    size_t min_status_samples = 3;
    size_t min_breathing_values = 1;
    std::chrono::seconds blink_min_window{10};
};

class AttentivenessScorer {
public:
    explicit AttentivenessScorer(const ScorerConfig& cfg = {}) : cfg_(cfg) {}

    void OnCoreMetrics(const presage::physiology::MetricsBuffer& metrics) {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();

        // Sample current face-presence state on each core-metrics tick so
        // presence scoring does not depend on sparse status-change events.
        face_presence_.push_back({now, current_face_ok_ ? 1.f : 0.f});

        for (int i = 0; i < metrics.face().blinking_size(); ++i) {
            const auto& ev = metrics.face().blinking(i);
            if (!ev.detected()) {
                continue;
            }

            const float t = ev.time();
            if (t <= blink_watermark_s_) {
                continue;
            }
            blink_watermark_s_ = t;

            if (!first_blink_wall_time_.has_value()) {
                first_blink_wall_time_ = now;
            }

            const auto wall_t = *first_blink_wall_time_ +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<float>(t));
            blink_events_.push_back(wall_t);
        }

        if (metrics.face().blinking_size() > 0) {
            const float last_t = metrics.face().blinking(metrics.face().blinking_size() - 1).time();
            measurement_duration_s_ = std::max(measurement_duration_s_, last_t);
        }

        const float br = metrics.breathing().strict().value();
        if (br > 0.f) {
            breathing_values_.push_back({now, br});
        }

        TrimToWindow(now);
    }

    void OnEdgeMetrics(const presage::physiology::Metrics&) {}

    void OnStatusChange(const presage::physiology::StatusValue& sv) {
        std::lock_guard<std::mutex> lock(mutex_);

        std::string description = presage::physiology::GetStatusDescription(sv.value());
        std::transform(description.begin(), description.end(), description.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        const bool face_not_visible =
            description.find("not visible") != std::string::npos ||
            description.find("no face") != std::string::npos;
        const bool too_much_movement =
            description.find("movement") != std::string::npos ||
            description.find("motion") != std::string::npos;
        const bool poor_lighting =
            description.find("poor lighting") != std::string::npos;

        current_face_ok_ = !(face_not_visible || too_much_movement || poor_lighting);
    }

    AttentivenessResult Compute() const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto now = std::chrono::steady_clock::now();

        AttentivenessResult result;
        result.timestamp = now;

        result.sub.presence_valid = (face_presence_.size() >= cfg_.min_status_samples);
        if (result.sub.presence_valid) {
            const float frac = Average(ValuesIn(face_presence_));
            result.sub.face_present_frac = frac;
            result.sub.presence_score = std::min(1.f, frac / cfg_.presence_full);
        }

        result.sub.breathing_valid = (breathing_values_.size() >= cfg_.min_breathing_values);
        if (result.sub.breathing_valid) {
            const float br = RecencyWeightedAverage(breathing_values_);
            result.sub.breathing_rate_bpm = br;
            result.sub.breathing_score = BreathRateCurve(br);
        }

        const float elapsed_s = std::min(
            measurement_duration_s_,
            static_cast<float>(cfg_.window.count()));

        result.sub.blink_valid =
            (elapsed_s >= static_cast<float>(cfg_.blink_min_window.count()));
        if (result.sub.blink_valid) {
            const auto window_start = now - cfg_.window;
            size_t n = 0;
            for (const auto& t : blink_events_) {
                if (t >= window_start) {
                    ++n;
                }
            }

            result.sub.blink_event_count = n;
            const float bpm = (elapsed_s > 0.f)
                                  ? (static_cast<float>(n) / elapsed_s) * 60.f
                                  : 0.f;
            result.sub.blink_rate_bpm = bpm;
            result.sub.blink_score = BlinkCurve(bpm);
        }

        float wsum = 0.f;
        float wtot = 0.f;
        auto add = [&](float score, float w, bool valid) {
            if (!valid) {
                return;
            }
            wsum += score * w;
            wtot += w;
        };

        add(result.sub.presence_score, cfg_.w_presence, result.sub.presence_valid);
        add(result.sub.breathing_score, cfg_.w_breathing, result.sub.breathing_valid);
        add(result.sub.blink_score, cfg_.w_blink, result.sub.blink_valid);

        if (wtot > 0.f) {
            result.score = std::clamp((wsum / wtot) * 100.f, 0.f, 100.f);
        }

        result.valid = result.sub.presence_valid &&
                       (result.sub.breathing_valid || result.sub.blink_valid);

        return result;
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        blink_events_.clear();
        breathing_values_.clear();
        face_presence_.clear();
        blink_watermark_s_ = -1.f;
        first_blink_wall_time_.reset();
        measurement_duration_s_ = 0.f;
    }

private:
    struct TimedSample {
        std::chrono::steady_clock::time_point time;
        float value;
    };

    float BreathRateCurve(float bpm) const {
        if (bpm < cfg_.breath_drowsy_lo) {
            return 0.1f;
        }
        if (bpm < cfg_.breath_alert_lo) {
            return Lerp(0.1f, 1.f, (bpm - cfg_.breath_drowsy_lo) /
                                  (cfg_.breath_alert_lo - cfg_.breath_drowsy_lo));
        }
        if (bpm <= cfg_.breath_alert_hi) {
            return 1.f;
        }
        if (bpm <= cfg_.breath_stressed) {
            return Lerp(1.f, 0.3f, (bpm - cfg_.breath_alert_hi) /
                                   (cfg_.breath_stressed - cfg_.breath_alert_hi));
        }
        return 0.3f;
    }

    float BlinkCurve(float bpm) const {
        if (bpm < cfg_.blink_very_low) {
            return Lerp(0.f, 0.4f, bpm / cfg_.blink_very_low);
        }
        if (bpm < cfg_.blink_focus_lo) {
            return Lerp(0.4f, 1.f, (bpm - cfg_.blink_very_low) /
                                   (cfg_.blink_focus_lo - cfg_.blink_very_low));
        }
        if (bpm <= cfg_.blink_focus_hi) {
            return 1.f;
        }
        if (bpm <= cfg_.blink_normal_hi) {
            return Lerp(1.f, 0.7f, (bpm - cfg_.blink_focus_hi) /
                                   (cfg_.blink_normal_hi - cfg_.blink_focus_hi));
        }
        if (bpm <= cfg_.blink_fatigue) {
            return Lerp(0.7f, 0.2f, (bpm - cfg_.blink_normal_hi) /
                                   (cfg_.blink_fatigue - cfg_.blink_normal_hi));
        }
        return 0.2f;
    }

    void TrimToWindow(std::chrono::steady_clock::time_point now) {
        const auto cutoff = now - cfg_.window;
        auto trim_deque = [&cutoff](std::deque<TimedSample>& d) {
            while (!d.empty() && d.front().time < cutoff) {
                d.pop_front();
            }
        };
        trim_deque(breathing_values_);
        trim_deque(face_presence_);
        blink_events_.erase(
            std::remove_if(blink_events_.begin(), blink_events_.end(),
                           [&](const auto& t) { return t < cutoff; }),
            blink_events_.end());
    }

    static float RecencyWeightedAverage(const std::deque<TimedSample>& d) {
        if (d.empty()) {
            return 0.f;
        }

        float wsum = 0.f;
        float wtot = 0.f;
        const float n = static_cast<float>(d.size());
        for (size_t i = 0; i < d.size(); ++i) {
            const float w = 1.f + static_cast<float>(i) / n;
            wsum += d[i].value * w;
            wtot += w;
        }
        return wsum / wtot;
    }

    static float Lerp(float a, float b, float t) {
        return a + std::clamp(t, 0.f, 1.f) * (b - a);
    }

    static float Average(const std::vector<float>& v) {
        if (v.empty()) {
            return 0.f;
        }
        return std::accumulate(v.begin(), v.end(), 0.f) / static_cast<float>(v.size());
    }

    static std::vector<float> ValuesIn(const std::deque<TimedSample>& d) {
        std::vector<float> v;
        v.reserve(d.size());
        for (const auto& s : d) {
            v.push_back(s.value);
        }
        return v;
    }

    ScorerConfig cfg_;
    mutable std::mutex mutex_;

    std::vector<std::chrono::steady_clock::time_point> blink_events_;
    float blink_watermark_s_ = -1.f;
    std::optional<std::chrono::steady_clock::time_point> first_blink_wall_time_;
    float measurement_duration_s_ = 0.f;

    bool current_face_ok_ = true;

    std::deque<TimedSample> breathing_values_;
    std::deque<TimedSample> face_presence_;
};

}  // namespace attention_demo
