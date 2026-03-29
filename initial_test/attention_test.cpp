// attention_example.cc
// Minimal attention estimator using SmartSpectraProcessor callbacks.
// NOTE: glue functions (EstimateHeadYawFromLandmarks, etc.) are basic heuristics and should be improved/calibrated.

#include "SmartSpectraProcessor.hpp"
#include <chrono>
#include <deque>
#include <numeric>
#include <optional>

// Simple exponential smoother
class EWMA {
public:
    EWMA(double alpha = 0.2) : alpha_(alpha) {}
    double Update(double value) {
        if (!initialized_) { val_ = value; initialized_ = true; }
        val_ = alpha_ * value + (1.0 - alpha_) * val_;
        return val_;
    }
    std::optional<double> Value() const { return initialized_ ? std::optional<double>(val_) : std::nullopt; }
private:
    double alpha_;
    double val_ = 0.0;
    bool initialized_ = false;
};

// Time series sliding window for event counting or values
template<typename T>
class SlidingWindow {
public:
    SlidingWindow(std::chrono::milliseconds window) : window_(window) {}
    void Push(const T& value, std::chrono::steady_clock::time_point when = std::chrono::steady_clock::now()) {
        data_.emplace_back(when, value);
        Trim();
    }
    std::vector<T> Values() const {
        std::vector<T> out; out.reserve(data_.size());
        for (auto& p : data_) out.push_back(p.second);
        return out;
    }
    size_t Count() const { return data_.size(); }
    void Trim() {
        auto cutoff = std::chrono::steady_clock::now() - window_;
        while (!data_.empty() && data_.front().first < cutoff) data_.pop_front();
    }
private:
    std::deque<std::pair<std::chrono::steady_clock::time_point, T>> data_;
    std::chrono::milliseconds window_;
};

// Heuristic: estimate yaw from 2D landmark positions (requires dense facemesh).
// This is a placeholder: replace with solvePnP-based pose estimation for better accuracy.
double EstimateHeadYawFromLandmarks(const std::vector<std::pair<float,float>>& landmarks) {
    // Expect landmarks to include left-eye and right-eye centers and nose tip.
    // Implementation note: landmark indices depend on facemesh convention (tune per SDK).
    if (landmarks.size() < 3) return 0.0;
    auto left_eye = landmarks[0];
    auto right_eye = landmarks[1];
    auto nose = landmarks[2];
    // Compare nose x position relative to eye midpoint:
    float eye_mid_x = 0.5f*(left_eye.first + right_eye.first);
    float dx = nose.first - eye_mid_x;
    // Normalize by approximate inter-eye distance
    float inter_eye = std::abs(right_eye.first - left_eye.first);
    if (inter_eye <= 1e-6f) return 0.0;
    double yaw = dx / inter_eye; // negative/positive indicates left/right
    // Clamp to [-1,1]
    if (yaw > 1.0) yaw = 1.0;
    if (yaw < -1.0) yaw = -1.0;
    return yaw; // -1..1
}

// Attention estimator class
class AttentionEstimator {
public:
    AttentionEstimator()
        : blink_window_(std::chrono::milliseconds(5000)), // 5s window
          visible_window_(std::chrono::milliseconds(3000)) // 3s window
    {}

    // Call this per-edge-metrics update
    double Update(const presage::physiology::Metrics& metrics) {
        auto now = std::chrono::steady_clock::now();

        // 1) Face visible
        bool face_visible = metrics.has_face() && metrics.face().has_visibility() ? metrics.face().visibility() : true;
        visible_window_.Push(face_visible ? 1 : 0, now);

        // 2) Blink events (store timestamps of blink detections)
        // MetricsBuffer example used blinking() event list; for edge metrics, adjust access as appropriate.
        for (const auto& b : metrics.face().blinking()) {
            // Each blink event may have time() or detected() fields; use detected() as presence signal.
            if (b.detected()) {
                blink_window_.Push(1, now);
            }
        }

        // 3) Talking flag (if user is speaking)
        bool talking = false;
        for (const auto& t : metrics.face().talking()) {
            if (t.detected()) { talking = true; break; }
        }

        // 4) Head yaw from landmarks (if dense facemesh available)
        double yaw = 0.0;
        std::vector<std::pair<float,float>> lm;
        if (metrics.face().has_mesh()) {
            // Build a tiny vector of key landmarks - this depends on SDK facemesh format.
            // Here we push 3 example points (left_eye_center, right_eye_center, nose_tip)
            // Replace indices with SDK's landmark indices
            for (int idx : {33, 263, 1}) { // example MediaPipe indices; verify with SDK
                if (idx < metrics.face().mesh().points_size()) {
                    const auto& p = metrics.face().mesh().points(idx);
                    lm.emplace_back(p.x(), p.y());
                }
            }
            if (lm.size() >= 3) yaw = EstimateHeadYawFromLandmarks(lm);
        }

        // Compute raw feature scores (0..1)
        double face_present_score = (visible_window_.Count() > 0) ? (static_cast<double>(visible_window_.Count()) / (visible_window_cap_)) : (face_visible ? 1.0 : 0.0);
        if (face_present_score > 1.0) face_present_score = 1.0;

        double gaze_score = 1.0 - std::min(1.0, std::abs(yaw)); // yaw near 0 -> high score
        double blink_rate = static_cast<double>(blink_window_.Count()) / (blink_window_seconds_); // blinks per sec
        // Ideal blink rate ~10-20/min = 0.17-0.33 Hz; map to 0..1 score favoring moderate rate
        double blink_score = 1.0 - std::min(1.0, std::abs(blink_rate - 0.25) / 0.25); // simple mapping

        // Head motion / stability: you can derive from consecutive landmark changes in a real implementation.
        double head_stability_score = 1.0; // placeholder (requires motion computation)

        // Talking penalty
        double talking_penalty = talking ? 0.7 : 0.0; // if talking, subtract 0.7 (tunable)

        // Weighted combination
        double raw_attention = 0.35*face_present_score + 0.40*gaze_score + 0.10*head_stability_score + 0.15*blink_score;
        raw_attention = raw_attention * (1.0 - talking_penalty);

        // Clamp and smooth
        if (raw_attention < 0) raw_attention = 0;
        if (raw_attention > 1) raw_attention = 1;

        // Smooth with EWMA
        double smoothed = smoother_.Update(raw_attention);
        return smoothed;
    }

private:
    SlidingWindow<int> blink_window_;
    SlidingWindow<int> visible_window_;
    EWMA smoother_{0.18};
    const double blink_window_seconds_ = 5.0;
    const double visible_window_cap_ = 3.0; // used for normalization of visibility fraction
};

int main() {
    // Set up your processor as in the primer
    my_app::SmartSpectraConfig cfg;
    cfg.api_key = "YOUR_API_KEY";
    cfg.enable_edge_metrics = true;
    cfg.enable_gui = false;
    my_app::SmartSpectraProcessor proc(cfg);

    AttentionEstimator estimator;

    proc.SetEdgeMetricsCallback([&estimator](const presage::physiology::Metrics& metrics) {
        double attention = estimator.Update(metrics);
        LOG(INFO) << "Attention score: " << attention;
        // You can send attention to GUI or downstream logic here.
        return absl::OkStatus();
    });

    // Initialize and start
    auto status = proc.Initialize();
    if (!status.ok()) { LOG(ERROR) << status.message(); return -1; }
    status = proc.Start();
    if (!status.ok()) { LOG(ERROR) << status.message(); return -1; }

    // For demo purposes let it run; in real app manage lifecycle and Stop()
    std::this_thread::sleep_for(std::chrono::minutes(10));
    proc.Stop();

    return 0;
}
