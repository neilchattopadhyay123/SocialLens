# SmartSpectra C++ SDK - Complete Implementation Guide

## Quick Start Guide

### Complete Working Example

```cpp
// main.cc - Minimal complete implementation
#include <smartspectra/container/foreground_container.hpp>
#include <smartspectra/container/settings.hpp>
#include <smartspectra/video_source/camera/camera.hpp>
#include <smartspectra/gui/opencv_hud.hpp>
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>
#include <opencv2/opencv.hpp>

namespace spectra = presage::smartspectra;
namespace settings = presage::smartspectra::container::settings;
namespace vs = presage::smartspectra::video_source;
using DeviceType = presage::platform_independence::DeviceType;

int main(int argc, char** argv) {
    // Initialize logging
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;
    
    // REQUIRED: Choose ONE authentication method
    
    // Option 1: API Key Authentication  
    std::string api_key = "YOUR_API_KEY_HERE";  // Get from https://physiology.presagetech.com
    
    // Option 2: OnPrem/gRPC Authentication (Enterprise only)
    // Configure gRPC settings for on-premise deployment
    
    // Configure settings for continuous monitoring
    settings::Settings<settings::OperationMode::Continuous, settings::IntegrationMode::Rest> settings{
        vs::VideoSourceSettings{
            0,    // camera_device_index
            vs::ResolutionSelectionMode::Auto,
            1280, // capture_width_px
            720,  // capture_height_px
            presage::camera::CameraResolutionRange::Unspecified_EnumEnd,
            presage::camera::CaptureCodec::MJPG,
            true, // auto_lock
            vs::InputTransformMode::Unspecified_EnumEnd,
            "",   // input_video_path (empty for camera)
            ""    // input_video_time_path
        },
        settings::VideoSinkSettings{},  // No video output
        false,  // headless mode
        20,     // interframe_delay_ms
        false,  // start_with_recording_on
        0,      // start_time_offset_ms
        true,   // scale_input
        true,   // binary_graph
        false,  // enable_phasic_bp
        false,  // enable_dense_facemesh_points
        false,  // use_full_range_face_detection
        false,  // use_full_pose_landmarks
        false,  // enable_pose_landmark_segmentation
        true,   // enable_edge_metrics
        false,  // print_graph_contents
        false,  // log_transfer_timing_info
        1,      // verbosity_level
        settings::ContinuousSettings{
            0.5   // preprocessed_data_buffer_duration_s
        },
        settings::RestSettings{
            api_key
        }
    };
    
    // Create container
    spectra::container::CpuContinuousRestForegroundContainer container(settings);
    
    // Set up HUD for real-time visualization
    spectra::gui::OpenCvHud hud(10, 0, 1260, 400);
    
    // Set up metrics callback
    auto status = container.SetOnCoreMetricsOutput(
        [&hud](const presage::physiology::MetricsBuffer& metrics, int64_t timestamp_microseconds) {
            // Convert to JSON for logging
            std::string metrics_json;
            google::protobuf::util::JsonPrintOptions options;
            options.add_whitespace = true;
            google::protobuf::util::MessageToJsonString(metrics, &metrics_json, options);
            
            LOG(INFO) << "Pulse: " << static_cast<int>(metrics.pulse().strict().value()) << " BPM, "
                      << "Breathing: " << static_cast<int>(metrics.breathing().strict().value()) << " BPM";
            
            // Update HUD with new metrics
            hud.UpdateWithNewMetrics(metrics);
            
            return absl::OkStatus();
        });
    
    // Set up video output callback for GUI
    if (status.ok()) {
        status = container.SetOnVideoOutput([&hud](cv::Mat& output_frame, int64_t timestamp_milliseconds) {
            auto render_status = hud.Render(output_frame);
            if (!render_status.ok()) {
                return render_status;
            }
            
            // Display the frame
            cv::imshow("SmartSpectra", output_frame);
            char key = cv::waitKey(1);
            if (key == 'q' || key == 27) {  // 'q' or ESC to quit
                return absl::CancelledError("User requested quit");
            }
            
            return absl::OkStatus();
        });
    }
    
    // Set up status change callback
    if (status.ok()) {
        status = container.SetOnStatusChange([](presage::physiology::StatusValue status) {
            LOG(INFO) << "Status: " << presage::physiology::GetStatusDescription(status.value());
            return absl::OkStatus();
        });
    }
    
    // Initialize and run
    if (status.ok()) { status = container.Initialize(); }
    if (status.ok()) { 
        LOG(INFO) << "Starting SmartSpectra processing. Press 's' to start recording, 'q' to quit.";
        status = container.Run(); 
    }
    
    cv::destroyAllWindows();
    
    if (!status.ok()) {
        LOG(ERROR) << "Run failed: " << status.message();
        return EXIT_FAILURE;
    } else {
        LOG(INFO) << "Success!";
        return EXIT_SUCCESS;
    }
}
```

### Step-by-Step Integration

1. **Install SDK Dependencies**
   ```bash
   # Ubuntu 22.04/Mint 21
   curl -s "https://presage-security.github.io/PPA/KEY.gpg" | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/presage-technologies.gpg >/dev/null
   sudo curl -s --compressed -o /etc/apt/sources.list.d/presage-technologies.list "https://presage-security.github.io/PPA/presage-technologies.list"
   sudo apt update
   sudo apt install libsmartspectra-dev
   ```

2. **Set Up CMakeLists.txt**
   ```cmake
   cmake_minimum_required(VERSION 3.27.0)
   project(MySmartSpectraApp CXX)
   
   find_package(SmartSpectra REQUIRED)
   find_package(OpenCV REQUIRED)
   find_package(PkgConfig REQUIRED)
   
   add_executable(my_app main.cc)
   
   target_link_libraries(my_app
       SmartSpectra::Container
       SmartSpectra::Gui
       ${OpenCV_LIBS}
   )
   ```

3. **Authentication Setup**
   - For API Key: Get key from https://physiology.presagetech.com
   - For OnPrem: Contact support for gRPC server deployment

4. **Build and Run**
   ```bash
   mkdir build && cd build
   cmake ..
   make
   ./my_app
   ```

## Core Implementation Components

### Container Management - ForegroundContainer.hpp

```cpp
// SmartSpectraProcessor.hpp
// Core processing container wrapper

#pragma once

#include <smartspectra/container/foreground_container.hpp>
#include <smartspectra/container/settings.hpp>
#include <smartspectra/gui/opencv_hud.hpp>
#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>

namespace my_app {

// Processing mode enumeration
enum class ProcessingMode {
    SPOT,        // Single measurement
    CONTINUOUS   // Continuous monitoring
};

// Authentication mode enumeration  
enum class AuthMode {
    API_KEY,     // REST API authentication
    GRPC_ONPREM  // On-premise gRPC
};

// Configuration structure
struct SmartSpectraConfig {
    ProcessingMode mode = ProcessingMode::CONTINUOUS;
    AuthMode auth_mode = AuthMode::API_KEY;
    std::string api_key;
    int camera_index = 0;
    int capture_width = 1280;
    int capture_height = 720;
    double spot_duration_s = 30.0;
    double buffer_duration_s = 0.5;
    bool enable_gui = true;
    bool enable_edge_metrics = true;
    bool enable_phasic_bp = false;
    int verbosity_level = 1;
    uint16_t grpc_port = 50051;
};

// Callback type definitions
using MetricsCallback = std::function<absl::Status(const presage::physiology::MetricsBuffer&, int64_t)>;
using EdgeMetricsCallback = std::function<absl::Status(const presage::physiology::Metrics&)>;
using VideoCallback = std::function<absl::Status(cv::Mat&, int64_t)>;
using StatusCallback = std::function<absl::Status(presage::physiology::StatusCode)>;

class SmartSpectraProcessor {
public:
    explicit SmartSpectraProcessor(const SmartSpectraConfig& config);
    ~SmartSpectraProcessor();
    
    // Configuration
    absl::Status Initialize();
    absl::Status Start();
    absl::Status Stop();
    bool IsRunning() const { return running_; }
    
    // Callback setters
    void SetMetricsCallback(MetricsCallback callback) { metrics_callback_ = std::move(callback); }
    void SetEdgeMetricsCallback(EdgeMetricsCallback callback) { edge_metrics_callback_ = std::move(callback); }
    void SetVideoCallback(VideoCallback callback) { video_callback_ = std::move(callback); }
    void SetStatusCallback(StatusCallback callback) { status_callback_ = std::move(callback); }

    // Recording control
    absl::Status StartRecording();
    absl::Status StopRecording();
    bool IsRecording() const;

    // Real-time access to latest metrics
    std::optional<presage::physiology::MetricsBuffer> GetLatestMetrics() const;
    std::optional<presage::physiology::Metrics> GetLatestEdgeMetrics() const;

private:
    SmartSpectraConfig config_;
    bool running_ = false;
    bool initialized_ = false;

    // Container instances for different modes
    std::unique_ptr<presage::smartspectra::container::CpuContinuousRestForegroundContainer> continuous_rest_container_;
    std::unique_ptr<presage::smartspectra::container::SpotRestForegroundContainer<presage::platform_independence::DeviceType::Cpu>> spot_rest_container_;
    std::unique_ptr<presage::smartspectra::container::CpuContinuousGrpcForegroundContainer> continuous_grpc_container_;
    std::unique_ptr<presage::smartspectra::container::SpotGrpcForegroundContainer<presage::platform_independence::DeviceType::Cpu>> spot_grpc_container_;

    // GUI components
    std::unique_ptr<presage::smartspectra::gui::OpenCvHud> hud_;
    std::unique_ptr<presage::smartspectra::gui::OpenCvTracePlotter> edge_plotter_;

    // Callbacks
    MetricsCallback metrics_callback_;
    EdgeMetricsCallback edge_metrics_callback_;
    VideoCallback video_callback_;
    StatusCallback status_callback_;

    // Latest data cache
    mutable std::mutex metrics_mutex_;
    std::optional<presage::physiology::MetricsBuffer> latest_metrics_;
    std::optional<presage::physiology::Metrics> latest_edge_metrics_;

    // Internal methods
    absl::Status CreateContainer();
    absl::Status SetupCallbacks();
    presage::smartspectra::container::settings::Settings<
        presage::smartspectra::container::settings::OperationMode::Continuous,
        presage::smartspectra::container::settings::IntegrationMode::Rest
    > CreateContinuousRestSettings();

    presage::smartspectra::container::settings::Settings<
        presage::smartspectra::container::settings::OperationMode::Spot,
        presage::smartspectra::container::settings::IntegrationMode::Rest
    > CreateSpotRestSettings();
};

} // namespace my_app
```

### SmartSpectraProcessor Implementation

```cpp
// SmartSpectraProcessor.cpp
// Implementation of the core processing wrapper

#include "SmartSpectraProcessor.hpp"
#include <glog/logging.h>
#include <google/protobuf/util/json_util.h>

namespace my_app {

SmartSpectraProcessor::SmartSpectraProcessor(const SmartSpectraConfig& config)
    : config_(config) {

    if (config_.enable_gui) {
        hud_ = std::make_unique<presage::smartspectra::gui::OpenCvHud>(10, 0, 1260, 400);

        if (config_.enable_edge_metrics) {
            edge_plotter_ = std::make_unique<presage::smartspectra::gui::OpenCvTracePlotter>(10, 450, 910, 100);
        }
    }
}

SmartSpectraProcessor::~SmartSpectraProcessor() {
    if (running_) {
        Stop();
    }
}

absl::Status SmartSpectraProcessor::Initialize() {
    if (initialized_) {
        return absl::AlreadyExistsError("Processor already initialized");
    }

    auto status = CreateContainer();
    if (!status.ok()) {
        return status;
    }

    status = SetupCallbacks();
    if (!status.ok()) {
        return status;
    }

    // Initialize the appropriate container
    if (config_.mode == ProcessingMode::CONTINUOUS) {
        if (config_.auth_mode == AuthMode::API_KEY && continuous_rest_container_) {
            status = continuous_rest_container_->Initialize();
        } else if (config_.auth_mode == AuthMode::GRPC_ONPREM && continuous_grpc_container_) {
            status = continuous_grpc_container_->Initialize();
        }
    } else if (config_.mode == ProcessingMode::SPOT) {
        if (config_.auth_mode == AuthMode::API_KEY && spot_rest_container_) {
            status = spot_rest_container_->Initialize();
        } else if (config_.auth_mode == AuthMode::GRPC_ONPREM && spot_grpc_container_) {
            status = spot_grpc_container_->Initialize();
        }
    }

    if (status.ok()) {
        initialized_ = true;
        LOG(INFO) << "SmartSpectra processor initialized successfully";
    }

    return status;
}

absl::Status SmartSpectraProcessor::Start() {
    if (!initialized_) {
        return absl::FailedPreconditionError("Processor not initialized");
    }

    if (running_) {
        return absl::AlreadyExistsError("Processor already running");
    }

    absl::Status status;

    // Start the appropriate container
    if (config_.mode == ProcessingMode::CONTINUOUS) {
        if (config_.auth_mode == AuthMode::API_KEY && continuous_rest_container_) {
            status = continuous_rest_container_->Run();
        } else if (config_.auth_mode == AuthMode::GRPC_ONPREM && continuous_grpc_container_) {
            status = continuous_grpc_container_->Run();
        }
    } else if (config_.mode == ProcessingMode::SPOT) {
        if (config_.auth_mode == AuthMode::API_KEY && spot_rest_container_) {
            status = spot_rest_container_->Run();
        } else if (config_.auth_mode == AuthMode::GRPC_ONPREM && spot_grpc_container_) {
            status = spot_grpc_container_->Run();
        }
    }

    if (status.ok()) {
        running_ = true;
        LOG(INFO) << "SmartSpectra processor started";
    }

    return status;
}

absl::Status SmartSpectraProcessor::Stop() {
    if (!running_) {
        return absl::OkStatus();  // Already stopped
    }

    // Stop containers - they handle cleanup internally
    running_ = false;
    LOG(INFO) << "SmartSpectra processor stopped";

    return absl::OkStatus();
}

absl::Status SmartSpectraProcessor::CreateContainer() {
    namespace settings = presage::smartspectra::container::settings;

    if (config_.mode == ProcessingMode::CONTINUOUS) {
        if (config_.auth_mode == AuthMode::API_KEY) {
            auto settings = CreateContinuousRestSettings();
            continuous_rest_container_ =
                std::make_unique<presage::smartspectra::container::CpuContinuousRestForegroundContainer>(settings);
        } else if (config_.auth_mode == AuthMode::GRPC_ONPREM) {
            // Create gRPC settings and container
            // Implementation would depend on gRPC container availability
        }
    } else if (config_.mode == ProcessingMode::SPOT) {
        if (config_.auth_mode == AuthMode::API_KEY) {
            auto settings = CreateSpotRestSettings();
            spot_rest_container_ =
                std::make_unique<presage::smartspectra::container::SpotRestForegroundContainer<presage::platform_independence::DeviceType::Cpu>>(settings);
        } else if (config_.auth_mode == AuthMode::GRPC_ONPREM) {
            // Create gRPC settings and container
        }
    }

    return absl::OkStatus();
}

absl::Status SmartSpectraProcessor::SetupCallbacks() {
    // Set up metrics callback
    auto metrics_handler = [this](const presage::physiology::MetricsBuffer& metrics, int64_t timestamp) {
        // Cache latest metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            latest_metrics_ = metrics;
        }

        // Update GUI if enabled
        if (hud_) {
            hud_->UpdateWithNewMetrics(metrics);
        }

        // Call user callback if set
        if (metrics_callback_) {
            return metrics_callback_(metrics, timestamp);
        }

        return absl::OkStatus();
    };

    // Set up edge metrics callback
    auto edge_metrics_handler = [this](const presage::physiology::Metrics& metrics) {
        // Cache latest edge metrics
        {
            std::lock_guard<std::mutex> lock(metrics_mutex_);
            latest_edge_metrics_ = metrics;
        }

        // Update edge plotter if enabled
        if (edge_plotter_) {
            const auto& upper_trace = metrics.breathing().upper_trace();
            if (!upper_trace.empty()) {
                edge_plotter_->UpdateTraceWithSample(*upper_trace.rbegin());
            }
        }

        // Call user callback if set
        if (edge_metrics_callback_) {
            return edge_metrics_callback_(metrics);
        }

        return absl::OkStatus();
    };

    // Set up video callback
    auto video_handler = [this](cv::Mat& frame, int64_t timestamp) {
        // Render GUI overlays
        if (hud_) {
            auto status = hud_->Render(frame);
            if (!status.ok()) {
                return status;
            }
        }

        if (edge_plotter_) {
            auto status = edge_plotter_->Render(frame, cv::Scalar(0, 165, 255));
            if (!status.ok()) {
                return status;
            }
        }

        // Call user callback if set
        if (video_callback_) {
            return video_callback_(frame, timestamp);
        }

        return absl::OkStatus();
    };

    // Set up status callback
    auto status_handler = [this](presage::physiology::StatusValue status) {
        if (status_callback_) {
            return status_callback_(status);
        }
        return absl::OkStatus();
    };

    // Apply callbacks to appropriate container
    absl::Status status;

    if (config_.mode == ProcessingMode::CONTINUOUS && config_.auth_mode == AuthMode::API_KEY && continuous_rest_container_) {
        status = continuous_rest_container_->SetOnCoreMetricsOutput(metrics_handler);
        if (status.ok() && config_.enable_edge_metrics) {
            status = continuous_rest_container_->SetOnEdgeMetricsOutput(edge_metrics_handler);
        }
        if (status.ok()) {
            status = continuous_rest_container_->SetOnVideoOutput(video_handler);
        }
        if (status.ok()) {
            status = continuous_rest_container_->SetOnStatusChange(status_handler);
        }
    } else if (config_.mode == ProcessingMode::SPOT && config_.auth_mode == AuthMode::API_KEY && spot_rest_container_) {
        status = spot_rest_container_->SetOnCoreMetricsOutput(metrics_handler);
        if (status.ok()) {
            status = spot_rest_container_->SetOnVideoOutput(video_handler);
        }
        if (status.ok()) {
            status = spot_rest_container_->SetOnStatusChange(status_handler);
        }
    }

    return status;
}

auto SmartSpectraProcessor::CreateContinuousRestSettings() ->
    presage::smartspectra::container::settings::Settings<
        presage::smartspectra::container::settings::OperationMode::Continuous,
        presage::smartspectra::container::settings::IntegrationMode::Rest
    > {

    namespace settings = presage::smartspectra::container::settings;
    namespace vs = presage::smartspectra::video_source;

    return settings::Settings<settings::OperationMode::Continuous, settings::IntegrationMode::Rest>{
        vs::VideoSourceSettings{
            config_.camera_index,
            vs::ResolutionSelectionMode::Auto,
            config_.capture_width,
            config_.capture_height,
            presage::camera::CameraResolutionRange::Unspecified_EnumEnd,
            presage::camera::CaptureCodec::MJPG,
            true,  // auto_lock
            vs::InputTransformMode::Unspecified_EnumEnd,
            "",    // input_video_path
            ""     // input_video_time_path
        },
        settings::VideoSinkSettings{},  // No video output by default
        !config_.enable_gui,            // headless
        20,                             // interframe_delay_ms
        false,                          // start_with_recording_on
        0,                              // start_time_offset_ms
        true,                           // scale_input
        true,                           // binary_graph
        config_.enable_phasic_bp,       // enable_phasic_bp
        false,                          // enable_dense_facemesh_points
        false,                          // use_full_range_face_detection
        false,                          // use_full_pose_landmarks
        false,                          // enable_pose_landmark_segmentation
        config_.enable_edge_metrics,    // enable_edge_metrics
        false,                          // print_graph_contents
        false,                          // log_transfer_timing_info
        config_.verbosity_level,        // verbosity_level
        settings::ContinuousSettings{
            config_.buffer_duration_s
        },
        settings::RestSettings{
            config_.api_key
        }
    };
}

auto SmartSpectraProcessor::CreateSpotRestSettings() ->
    presage::smartspectra::container::settings::Settings<
        presage::smartspectra::container::settings::OperationMode::Spot,
        presage::smartspectra::container::settings::IntegrationMode::Rest
    > {

    namespace settings = presage::smartspectra::container::settings;
    namespace vs = presage::smartspectra::video_source;

    return settings::Settings<settings::OperationMode::Spot, settings::IntegrationMode::Rest>{
        vs::VideoSourceSettings{
            config_.camera_index,
            vs::ResolutionSelectionMode::Auto,
            config_.capture_width,
            config_.capture_height,
            presage::camera::CameraResolutionRange::Unspecified_EnumEnd,
            presage::camera::CaptureCodec::MJPG,
            true,  // auto_lock
            vs::InputTransformMode::Unspecified_EnumEnd,
            "",    // input_video_path
            ""     // input_video_time_path
        },
        settings::VideoSinkSettings{},  // No video output by default
        !config_.enable_gui,            // headless
        20,                             // interframe_delay_ms
        false,                          // start_with_recording_on
        0,                              // start_time_offset_ms
        true,                           // scale_input
        true,                           // binary_graph
        config_.enable_phasic_bp,       // enable_phasic_bp
        false,                          // enable_dense_facemesh_points
        false,                          // use_full_range_face_detection
        false,                          // use_full_pose_landmarks
        false,                          // enable_pose_landmark_segmentation
        false,                          // enable_edge_metrics (not available in spot mode)
        false,                          // print_graph_contents
        false,                          // log_transfer_timing_info
        config_.verbosity_level,        // verbosity_level
        settings::SpotSettings{
            config_.spot_duration_s
        },
        settings::RestSettings{
            config_.api_key
        }
    };
}

std::optional<presage::physiology::MetricsBuffer> SmartSpectraProcessor::GetLatestMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return latest_metrics_;
}

std::optional<presage::physiology::Metrics> SmartSpectraProcessor::GetLatestEdgeMetrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return latest_edge_metrics_;
}

} // namespace my_app
```

### GUI Integration - HUD and Visualization

```cpp
// SmartSpectraGUI.hpp
// GUI wrapper for OpenCV-based visualization

#pragma once

#include <smartspectra/gui/opencv_hud.hpp>
#include <smartspectra/gui/opencv_trace_plotter.hpp>
#include <smartspectra/gui/opencv_label.hpp>
#include <smartspectra/gui/opencv_value_indicator.hpp>
#include <opencv2/opencv.hpp>
#include <memory>

namespace my_app {

class SmartSpectraGUI {
public:
    struct Config {
        bool enable_hud = true;
        bool enable_edge_metrics = true;
        bool enable_performance_metrics = false;
        bool enable_face_mesh = false;
        cv::Size window_size = cv::Size(1280, 720);
        std::string window_name = "SmartSpectra";
    };

    explicit SmartSpectraGUI(const Config& config = {});
    ~SmartSpectraGUI();

    // Main rendering method
    absl::Status RenderFrame(cv::Mat& frame, int64_t timestamp_ms);

    // Data update methods
    absl::Status UpdateMetrics(const presage::physiology::MetricsBuffer& metrics);
    absl::Status UpdateEdgeMetrics(const presage::physiology::Metrics& metrics);
    absl::Status UpdatePerformanceMetrics(double fps, double latency_seconds);
    absl::Status UpdateFaceMesh(const std::vector<std::pair<float, float>>& points);

    // Display control
    void Show();
    void Hide();
    bool IsVisible() const { return visible_; }

    // Event handling
    char WaitKey(int delay_ms = 1);
    bool ShouldClose() const { return should_close_; }

private:
    Config config_;
    bool visible_ = false;
    bool should_close_ = false;

    // GUI components
    std::unique_ptr<presage::smartspectra::gui::OpenCvHud> hud_;
    std::unique_ptr<presage::smartspectra::gui::OpenCvTracePlotter> edge_plotter_;
    std::unique_ptr<presage::smartspectra::gui::OpenCvLabel> edge_label_;
    std::unique_ptr<presage::smartspectra::gui::OpenCvValueIndicator> fps_indicator_;
    std::unique_ptr<presage::smartspectra::gui::OpenCvLabel> fps_label_;
    std::unique_ptr<presage::smartspectra::gui::OpenCvValueIndicator> latency_indicator_;
    std::unique_ptr<presage::smartspectra::gui::OpenCvLabel> latency_label_;

    // Performance tracking
    double current_fps_ = 0.0;
    double current_latency_ = 0.0;

    // Face mesh points
    std::vector<std::pair<float, float>> face_mesh_points_;

    void InitializeComponents();
    absl::Status RenderFaceMesh(cv::Mat& frame);
};

} // namespace my_app
```

### Data Processing and Analytics

```cpp
// MetricsAnalyzer.hpp
// Advanced metrics processing and analysis

#pragma once

#include <physiology/modules/messages/metrics.h>
#include <vector>
#include <deque>
#include <optional>
#include <chrono>
#include <functional>

namespace my_app {

// Statistical metrics
struct VitalStatistics {
    double mean = 0.0;
    double std_dev = 0.0;
    double min_value = 0.0;
    double max_value = 0.0;
    double median = 0.0;
    size_t sample_count = 0;
};

// Time-series data point
struct TimeSeriesPoint {
    std::chrono::steady_clock::time_point timestamp;
    double value;
    double confidence = 1.0;
};

// Anomaly detection result
struct AnomalyResult {
    bool is_anomaly = false;
    double confidence = 0.0;
    std::string description;
};

class MetricsAnalyzer {
public:
    struct Config {
        size_t max_history_size = 1000;
        double anomaly_threshold = 2.0;  // Standard deviations
        std::chrono::seconds analysis_window = std::chrono::seconds(60);
        bool enable_smoothing = true;
        double smoothing_factor = 0.1;  // For exponential smoothing
    };

    explicit MetricsAnalyzer(const Config& config = {});

    // Data ingestion
    void AddMetrics(const presage::physiology::MetricsBuffer& metrics);
    void AddEdgeMetrics(const presage::physiology::Metrics& metrics);

    // Statistics calculation
    VitalStatistics GetPulseStatistics(std::chrono::seconds window = std::chrono::seconds(60)) const;
    VitalStatistics GetBreathingStatistics(std::chrono::seconds window = std::chrono::seconds(60)) const;

    // Real-time analysis
    std::optional<double> GetSmoothedPulseRate() const;
    std::optional<double> GetSmoothedBreathingRate() const;
    AnomalyResult DetectPulseAnomaly(double current_pulse) const;
    AnomalyResult DetectBreathingAnomaly(double current_breathing) const;

    // Data export
    std::vector<TimeSeriesPoint> GetPulseTimeSeries(std::chrono::seconds window = std::chrono::seconds(300)) const;
    std::vector<TimeSeriesPoint> GetBreathingTimeSeries(std::chrono::seconds window = std::chrono::seconds(300)) const;

    // Health insights
    struct HealthInsights {
        std::optional<double> average_pulse_bpm;
        std::optional<double> average_breathing_bpm;
        std::optional<double> pulse_variability;
        std::optional<double> breathing_variability;
        std::vector<std::string> recommendations;
    };

    HealthInsights GenerateHealthInsights() const;

    // Data persistence
    absl::Status SaveToJson(const std::string& filepath) const;
    absl::Status LoadFromJson(const std::string& filepath);

    // Callbacks for real-time alerts
    using AlertCallback = std::function<void(const AnomalyResult&, const std::string& vital_type)>;
    void SetAlertCallback(AlertCallback callback) { alert_callback_ = std::move(callback); }

private:
    Config config_;

    // Time series storage
    std::deque<TimeSeriesPoint> pulse_history_;
    std::deque<TimeSeriesPoint> breathing_history_;

    // Smoothed values
    mutable std::optional<double> smoothed_pulse_;
    mutable std::optional<double> smoothed_breathing_;

    // Alert callback
    AlertCallback alert_callback_;

    // Helper methods
    VitalStatistics CalculateStatistics(const std::deque<TimeSeriesPoint>& data, std::chrono::seconds window) const;
    std::vector<TimeSeriesPoint> GetTimeSeriesWindow(const std::deque<TimeSeriesPoint>& data, std::chrono::seconds window) const;
    void TrimHistory();
    double ApplyExponentialSmoothing(double new_value, std::optional<double>& smoothed_value) const;
};

} // namespace my_app
```

## Complete Usage Examples

### Advanced Continuous Monitoring Application

```cpp
// advanced_continuous_app.cc
// Comprehensive continuous monitoring with analytics

#include "SmartSpectraProcessor.hpp"
#include "SmartSpectraGUI.hpp"
#include "MetricsAnalyzer.hpp"
#include <glog/logging.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <fstream>
#include <iomanip>
#include <thread>

// Command line flags
ABSL_FLAG(std::string, api_key, "", "API key for Physiology service");
ABSL_FLAG(int, camera_index, 0, "Camera device index");
ABSL_FLAG(bool, enable_gui, true, "Enable GUI display");
ABSL_FLAG(bool, enable_analytics, true, "Enable advanced analytics");
ABSL_FLAG(std::string, output_dir, "output", "Output directory for data");
ABSL_FLAG(int, duration_minutes, 0, "Run duration in minutes (0 = infinite)");

class AdvancedMonitoringApp {
public:
    AdvancedMonitoringApp() {
        // Configure processor
        my_app::SmartSpectraConfig config;
        config.api_key = absl::GetFlag(FLAGS_api_key);
        config.camera_index = absl::GetFlag(FLAGS_camera_index);
        config.mode = my_app::ProcessingMode::CONTINUOUS;
        config.enable_gui = absl::GetFlag(FLAGS_enable_gui);
        config.enable_edge_metrics = true;
        config.verbosity_level = 1;

        processor_ = std::make_unique<my_app::SmartSpectraProcessor>(config);

        // Configure GUI if enabled
        if (config.enable_gui) {
            my_app::SmartSpectraGUI::Config gui_config;
            gui_config.enable_hud = true;
            gui_config.enable_edge_metrics = true;
            gui_config.enable_performance_metrics = true;
            gui_ = std::make_unique<my_app::SmartSpectraGUI>(gui_config);
        }

        // Configure analytics if enabled
        if (absl::GetFlag(FLAGS_enable_analytics)) {
            my_app::MetricsAnalyzer::Config analytics_config;
            analytics_config.max_history_size = 2000;
            analytics_config.enable_smoothing = true;
            analyzer_ = std::make_unique<my_app::MetricsAnalyzer>(analytics_config);

            // Set up anomaly alerts
            analyzer_->SetAlertCallback([this](const my_app::AnomalyResult& anomaly, const std::string& vital_type) {
                if (anomaly.is_anomaly) {
                    LOG(WARNING) << "Anomaly detected in " << vital_type << ": " << anomaly.description
                                 << " (confidence: " << std::fixed << std::setprecision(2) << anomaly.confidence << ")";
                }
            });
        }

        SetupCallbacks();
    }

    absl::Status Run() {
        LOG(INFO) << "Initializing SmartSpectra processor...";
        auto status = processor_->Initialize();
        if (!status.ok()) {
            return status;
        }

        LOG(INFO) << "Starting continuous monitoring...";
        LOG(INFO) << "Press 's' to start recording, 'q' to quit, 'a' to show analytics";

        if (gui_) {
            gui_->Show();
        }

        // Start processor in separate thread
        std::thread processor_thread([this]() {
            auto status = processor_->Start();
            if (!status.ok()) {
                LOG(ERROR) << "Processor failed: " << status.message();
            }
        });

        // Main event loop
        auto start_time = std::chrono::steady_clock::now();
        int duration_minutes = absl::GetFlag(FLAGS_duration_minutes);

        while (!should_quit_) {
            // Check duration limit
            if (duration_minutes > 0) {
                auto elapsed = std::chrono::steady_clock::now() - start_time;
                if (elapsed >= std::chrono::minutes(duration_minutes)) {
                    LOG(INFO) << "Duration limit reached, stopping...";
                    break;
                }
            }

            // Handle GUI events
            if (gui_) {
                char key = gui_->WaitKey(30);
                HandleKeyPress(key);

                if (gui_->ShouldClose()) {
                    break;
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // Cleanup
        processor_->Stop();
        processor_thread.join();

        // Save analytics data
        if (analyzer_) {
            SaveAnalyticsData();
            PrintHealthInsights();
        }

        return absl::OkStatus();
    }

private:
    std::unique_ptr<my_app::SmartSpectraProcessor> processor_;
    std::unique_ptr<my_app::SmartSpectraGUI> gui_;
    std::unique_ptr<my_app::MetricsAnalyzer> analyzer_;
    bool should_quit_ = false;
    bool recording_ = false;

    void SetupCallbacks() {
        // Metrics callback
        processor_->SetMetricsCallback([this](const presage::physiology::MetricsBuffer& metrics, int64_t timestamp) {
            if (analyzer_) {
                analyzer_->AddMetrics(metrics);
            }

            if (gui_) {
                gui_->UpdateMetrics(metrics);
            }

            // Log current vitals
            int pulse = static_cast<int>(metrics.pulse().strict().value());
            int breathing = static_cast<int>(metrics.breathing().strict().value());

            LOG(INFO) << "Vitals - Pulse: " << pulse << " BPM, Breathing: " << breathing << " BPM";

            return absl::OkStatus();
        });

        // Edge metrics callback
        processor_->SetEdgeMetricsCallback([this](const presage::physiology::Metrics& metrics) {
            if (analyzer_) {
                analyzer_->AddEdgeMetrics(metrics);
            }

            if (gui_) {
                gui_->UpdateEdgeMetrics(metrics);
            }

            return absl::OkStatus();
        });

        // Video callback
        processor_->SetVideoCallback([this](cv::Mat& frame, int64_t timestamp) {
            if (gui_) {
                return gui_->RenderFrame(frame, timestamp);
            }
            return absl::OkStatus();
        });

        // Status callback
        processor_->SetStatusCallback([](presage::physiology::StatusCode status) {
            LOG(INFO) << "Status: " << presage::physiology::GetStatusDescription(status);
            return absl::OkStatus();
        });
    }

    void HandleKeyPress(char key) {
        switch (key) {
            case 'q':
            case 27:  // ESC
                should_quit_ = true;
                break;

            case 's':
                if (!recording_) {
                    LOG(INFO) << "Starting measurement recording...";
                    processor_->StartRecording();
                    recording_ = true;
                } else {
                    LOG(INFO) << "Stopping measurement recording...";
                    processor_->StopRecording();
                    recording_ = false;
                }
                break;

            case 'a':
                if (analyzer_) {
                    PrintCurrentAnalytics();
                }
                break;

            case 'r':
                LOG(INFO) << "Resetting analytics history...";
                if (analyzer_) {
                    analyzer_ = std::make_unique<my_app::MetricsAnalyzer>();
                }
                break;
        }
    }

    void PrintCurrentAnalytics() {
        if (!analyzer_) return;

        auto pulse_stats = analyzer_->GetPulseStatistics();
        auto breathing_stats = analyzer_->GetBreathingStatistics();

        LOG(INFO) << "=== Current Analytics ===";
        LOG(INFO) << "Pulse - Mean: " << std::fixed << std::setprecision(1) << pulse_stats.mean
                  << " BPM, StdDev: " << pulse_stats.std_dev
                  << ", Range: [" << pulse_stats.min_value << ", " << pulse_stats.max_value << "]";
        LOG(INFO) << "Breathing - Mean: " << breathing_stats.mean
                  << " BPM, StdDev: " << breathing_stats.std_dev
                  << ", Range: [" << breathing_stats.min_value << ", " << breathing_stats.max_value << "]";

        if (auto smoothed_pulse = analyzer_->GetSmoothedPulseRate()) {
            LOG(INFO) << "Smoothed Pulse: " << *smoothed_pulse << " BPM";
        }

        if (auto smoothed_breathing = analyzer_->GetSmoothedBreathingRate()) {
            LOG(INFO) << "Smoothed Breathing: " << *smoothed_breathing << " BPM";
        }
    }

    void SaveAnalyticsData() {
        std::string output_dir = absl::GetFlag(FLAGS_output_dir);

        // Create output directory
        std::filesystem::create_directories(output_dir);

        // Save analytics to JSON
        std::string json_path = output_dir + "/analytics.json";
        auto status = analyzer_->SaveToJson(json_path);
        if (status.ok()) {
            LOG(INFO) << "Analytics data saved to: " << json_path;
        } else {
            LOG(ERROR) << "Failed to save analytics: " << status.message();
        }

        // Save time series data as CSV
        SaveTimeSeriesCSV(output_dir + "/pulse_timeseries.csv",
                         analyzer_->GetPulseTimeSeries(std::chrono::minutes(30)));
        SaveTimeSeriesCSV(output_dir + "/breathing_timeseries.csv",
                         analyzer_->GetBreathingTimeSeries(std::chrono::minutes(30)));
    }

    void SaveTimeSeriesCSV(const std::string& filepath, const std::vector<my_app::TimeSeriesPoint>& data) {
        std::ofstream file(filepath);
        file << "timestamp,value,confidence\n";

        for (const auto& point : data) {
            auto time_t = std::chrono::steady_clock::to_time_t(point.timestamp);
            file << time_t << "," << point.value << "," << point.confidence << "\n";
        }

        LOG(INFO) << "Time series data saved to: " << filepath;
    }

    void PrintHealthInsights() {
        if (!analyzer_) return;

        auto insights = analyzer_->GenerateHealthInsights();

        LOG(INFO) << "=== Health Insights ===";
        if (insights.average_pulse_bpm) {
            LOG(INFO) << "Average Pulse Rate: " << std::fixed << std::setprecision(1)
                      << *insights.average_pulse_bpm << " BPM";
        }

        if (insights.average_breathing_bpm) {
            LOG(INFO) << "Average Breathing Rate: " << *insights.average_breathing_bpm << " BPM";
        }

        if (insights.pulse_variability) {
            LOG(INFO) << "Pulse Variability: " << *insights.pulse_variability;
        }

        if (insights.breathing_variability) {
            LOG(INFO) << "Breathing Variability: " << *insights.breathing_variability;
        }

        for (const auto& recommendation : insights.recommendations) {
            LOG(INFO) << "Recommendation: " << recommendation;
        }
    }
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    absl::SetProgramUsageMessage(
        "Advanced SmartSpectra continuous monitoring application with analytics.\n"
        "Usage: ./advanced_continuous_app --api_key=YOUR_KEY [options]"
    );
    absl::ParseCommandLine(argc, argv);

    if (absl::GetFlag(FLAGS_api_key).empty()) {
        LOG(ERROR) << "API key is required. Use --api_key=YOUR_KEY";
        return EXIT_FAILURE;
    }

    AdvancedMonitoringApp app;
    auto status = app.Run();

    if (!status.ok()) {
        LOG(ERROR) << "Application failed: " << status.message();
        return EXIT_FAILURE;
    }

    LOG(INFO) << "Application completed successfully";
    return EXIT_SUCCESS;
}
```

### Spot Measurement Application

```cpp
// spot_measurement_app.cc
// Single measurement application with detailed results

#include "SmartSpectraProcessor.hpp"
#include <glog/logging.h>
#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <google/protobuf/util/json_util.h>
#include <fstream>
#include <iomanip>

// Command line flags
ABSL_FLAG(std::string, api_key, "", "API key for Physiology service");
ABSL_FLAG(int, camera_index, 0, "Camera device index");
ABSL_FLAG(double, duration, 30.0, "Measurement duration in seconds");
ABSL_FLAG(std::string, output_file, "", "Output file for results (optional)");
ABSL_FLAG(bool, verbose, false, "Verbose output");

class SpotMeasurementApp {
public:
    SpotMeasurementApp() {
        my_app::SmartSpectraConfig config;
        config.api_key = absl::GetFlag(FLAGS_api_key);
        config.camera_index = absl::GetFlag(FLAGS_camera_index);
        config.mode = my_app::ProcessingMode::SPOT;
        config.spot_duration_s = absl::GetFlag(FLAGS_duration);
        config.enable_gui = true;
        config.verbosity_level = absl::GetFlag(FLAGS_verbose) ? 2 : 1;

        processor_ = std::make_unique<my_app::SmartSpectraProcessor>(config);
        SetupCallbacks();
    }

    absl::Status Run() {
        LOG(INFO) << "Initializing SmartSpectra for spot measurement...";
        auto status = processor_->Initialize();
        if (!status.ok()) {
            return status;
        }

        LOG(INFO) << "Starting " << absl::GetFlag(FLAGS_duration) << "s measurement...";
        LOG(INFO) << "Please position your face in the camera and remain still.";

        measurement_complete_ = false;

        // Start measurement
        status = processor_->Start();
        if (!status.ok()) {
            return status;
        }

        // Wait for completion
        while (!measurement_complete_ && !should_quit_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        processor_->Stop();

        if (final_metrics_) {
            DisplayResults(*final_metrics_);

            if (!absl::GetFlag(FLAGS_output_file).empty()) {
                SaveResults(*final_metrics_);
            }
        } else {
            LOG(WARNING) << "No final metrics received";
        }

        return absl::OkStatus();
    }

private:
    std::unique_ptr<my_app::SmartSpectraProcessor> processor_;
    bool measurement_complete_ = false;
    bool should_quit_ = false;
    std::optional<presage::physiology::MetricsBuffer> final_metrics_;

    void SetupCallbacks() {
        processor_->SetMetricsCallback([this](const presage::physiology::MetricsBuffer& metrics, int64_t timestamp) {
            LOG(INFO) << "Measurement complete!";
            final_metrics_ = metrics;
            measurement_complete_ = true;
            return absl::OkStatus();
        });

        processor_->SetVideoCallback([this](cv::Mat& frame, int64_t timestamp) {
            cv::imshow("SmartSpectra Spot Measurement", frame);
            char key = cv::waitKey(1);
            if (key == 'q' || key == 27) {
                should_quit_ = true;
            }
            return absl::OkStatus();
        });

        processor_->SetStatusCallback([](presage::physiology::StatusCode status) {
            std::string status_msg = presage::physiology::GetStatusDescription(status);
            LOG(INFO) << "Status: " << status_msg;
            return absl::OkStatus();
        });
    }

    void DisplayResults(const presage::physiology::MetricsBuffer& metrics) {
        LOG(INFO) << "=== Measurement Results ===";

        // Basic vitals
        int pulse_rate = static_cast<int>(metrics.pulse().strict().value());
        int breathing_rate = static_cast<int>(metrics.breathing().strict().value());

        LOG(INFO) << "Pulse Rate: " << pulse_rate << " BPM";
        LOG(INFO) << "Breathing Rate: " << breathing_rate << " BPM";

        // Detailed metrics if verbose
        if (absl::GetFlag(FLAGS_verbose)) {
            LOG(INFO) << "=== Detailed Metrics ===";

            // Pulse details
            if (metrics.pulse().rate_size() > 0) {
                const auto& last_pulse = metrics.pulse().rate(metrics.pulse().rate_size() - 1);
                LOG(INFO) << "Final Pulse Confidence: " << std::fixed << std::setprecision(2)
                          << last_pulse.confidence();
            }

            // Breathing details
            if (metrics.breathing().rate_size() > 0) {
                const auto& last_breathing = metrics.breathing().rate(metrics.breathing().rate_size() - 1);
                LOG(INFO) << "Final Breathing Confidence: " << std::fixed << std::setprecision(2)
                          << last_breathing.confidence();
            }

            // Blood pressure (if available)
            if (metrics.has_blood_pressure() && metrics.blood_pressure().phasic_size() > 0) {
                const auto& phasic = metrics.blood_pressure().phasic(metrics.blood_pressure().phasic_size() - 1);
                LOG(INFO) << "Phasic Blood Pressure: " << std::fixed << std::setprecision(1)
                          << phasic.value();
            }

            // Metadata
            if (metrics.has_metadata()) {
                LOG(INFO) << "Measurement ID: " << metrics.metadata().id();
                LOG(INFO) << "Upload Timestamp: " << metrics.metadata().upload_timestamp();
            }
        }

        // Health assessment
        ProvideHealthAssessment(pulse_rate, breathing_rate);
    }

    void ProvideHealthAssessment(int pulse_rate, int breathing_rate) {
        LOG(INFO) << "=== Health Assessment ===";

        // Pulse rate assessment
        if (pulse_rate >= 60 && pulse_rate <= 100) {
            LOG(INFO) << "Pulse rate is within normal resting range (60-100 BPM)";
        } else if (pulse_rate < 60) {
            LOG(INFO) << "Pulse rate is below normal range (bradycardia)";
        } else if (pulse_rate > 100) {
            LOG(INFO) << "Pulse rate is above normal range (tachycardia)";
        }

        // Breathing rate assessment
        if (breathing_rate >= 12 && breathing_rate <= 20) {
            LOG(INFO) << "Breathing rate is within normal range (12-20 BPM)";
        } else if (breathing_rate < 12) {
            LOG(INFO) << "Breathing rate is below normal range";
        } else if (breathing_rate > 20) {
            LOG(INFO) << "Breathing rate is above normal range";
        }

        LOG(INFO) << "Note: These are general guidelines. Consult a healthcare professional for medical advice.";
    }

    void SaveResults(const presage::physiology::MetricsBuffer& metrics) {
        std::string output_file = absl::GetFlag(FLAGS_output_file);

        // Convert to JSON
        std::string metrics_json;
        google::protobuf::util::JsonPrintOptions options;
        options.add_whitespace = true;
        options.preserve_proto_field_names = true;

        auto status = google::protobuf::util::MessageToJsonString(metrics, &metrics_json, options);
        if (!status.ok()) {
            LOG(ERROR) << "Failed to convert metrics to JSON: " << status.message();
            return;
        }

        // Save to file
        std::ofstream file(output_file);
        if (!file.is_open()) {
            LOG(ERROR) << "Failed to open output file: " << output_file;
            return;
        }

        file << metrics_json;
        file.close();

        LOG(INFO) << "Results saved to: " << output_file;
    }
};

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    absl::SetProgramUsageMessage(
        "SmartSpectra spot measurement application.\n"
        "Performs a single measurement and displays detailed results.\n"
        "Usage: ./spot_measurement_app --api_key=YOUR_KEY [options]"
    );
    absl::ParseCommandLine(argc, argv);

    if (absl::GetFlag(FLAGS_api_key).empty()) {
        LOG(ERROR) << "API key is required. Use --api_key=YOUR_KEY";
        return EXIT_FAILURE;
    }

    double duration = absl::GetFlag(FLAGS_duration);
    if (duration < 20.0 || duration > 120.0) {
        LOG(ERROR) << "Duration must be between 20 and 120 seconds";
        return EXIT_FAILURE;
    }

    SpotMeasurementApp app;
    auto status = app.Run();

    cv::destroyAllWindows();

    if (!status.ok()) {
        LOG(ERROR) << "Application failed: " << status.message();
        return EXIT_FAILURE;
    }

    LOG(INFO) << "Measurement completed successfully";
    return EXIT_SUCCESS;
}
```

## API Reference with Code Snippets

### Core Container Classes

```cpp
// Container types for different operation modes and integration types

// Continuous mode with REST API
using ContinuousRestContainer = presage::smartspectra::container::CpuContinuousRestForegroundContainer;

// Spot mode with REST API
using SpotRestContainer = presage::smartspectra::container::SpotRestForegroundContainer<presage::platform_independence::DeviceType::Cpu>;

// Continuous mode with gRPC (OnPrem)
using ContinuousGrpcContainer = presage::smartspectra::container::CpuContinuousGrpcForegroundContainer;

// Spot mode with gRPC (OnPrem)
using SpotGrpcContainer = presage::smartspectra::container::SpotGrpcForegroundContainer<presage::platform_independence::DeviceType::Cpu>;

// Container initialization and control
auto container = std::make_unique<ContinuousRestContainer>(settings);
MP_RETURN_IF_ERROR(container->Initialize());
MP_RETURN_IF_ERROR(container->Run());  // Blocking call
```

### Settings Configuration

```cpp
// Settings for Continuous + REST mode
presage::smartspectra::container::settings::Settings<
    presage::smartspectra::container::settings::OperationMode::Continuous,
    presage::smartspectra::container::settings::IntegrationMode::Rest
> settings{
    // Video source settings
    presage::smartspectra::video_source::VideoSourceSettings{
        0,    // camera_device_index
        presage::smartspectra::video_source::ResolutionSelectionMode::Auto,
        1280, // capture_width_px
        720,  // capture_height_px
        presage::camera::CameraResolutionRange::Unspecified_EnumEnd,
        presage::camera::CaptureCodec::MJPG,
        true, // auto_lock
        presage::smartspectra::video_source::InputTransformMode::Unspecified_EnumEnd,
        "",   // input_video_path (empty for camera)
        ""    // input_video_time_path
    },
    // Video sink settings (for output)
    presage::smartspectra::container::settings::VideoSinkSettings{},
    false,  // headless
    20,     // interframe_delay_ms
    false,  // start_with_recording_on
    0,      // start_time_offset_ms
    true,   // scale_input
    true,   // binary_graph
    false,  // enable_phasic_bp
    false,  // enable_dense_facemesh_points
    false,  // use_full_range_face_detection
    false,  // use_full_pose_landmarks
    false,  // enable_pose_landmark_segmentation
    true,   // enable_edge_metrics
    false,  // print_graph_contents
    false,  // log_transfer_timing_info
    1,      // verbosity_level
    // Mode-specific settings
    presage::smartspectra::container::settings::ContinuousSettings{
        0.5   // preprocessed_data_buffer_duration_s
    },
    // Integration-specific settings
    presage::smartspectra::container::settings::RestSettings{
        "YOUR_API_KEY_HERE"
    }
};
```

### Callback Registration

```cpp
// Metrics callback (main results from server)
MP_RETURN_IF_ERROR(container->SetOnCoreMetricsOutput(
    [](const presage::physiology::MetricsBuffer& metrics, int64_t timestamp_microseconds) {
        // Access pulse data
        int pulse_rate = static_cast<int>(metrics.pulse().strict().value());
        float pulse_confidence = 0.0f;
        if (metrics.pulse().rate_size() > 0) {
            pulse_confidence = metrics.pulse().rate(metrics.pulse().rate_size() - 1).confidence();
        }

        // Access breathing data
        int breathing_rate = static_cast<int>(metrics.breathing().strict().value());
        float breathing_confidence = 0.0f;
        if (metrics.breathing().rate_size() > 0) {
            breathing_confidence = metrics.breathing().rate(metrics.breathing().rate_size() - 1).confidence();
        }

        // Access time series data
        for (const auto& measurement : metrics.pulse().trace()) {
            float time = measurement.time();
            float value = measurement.value();
            // Process pulse trace data
        }

        for (const auto& measurement : metrics.breathing().upper_trace()) {
            float time = measurement.time();
            float value = measurement.value();
            // Process breathing trace data
        }

        LOG(INFO) << "Pulse: " << pulse_rate << " BPM (" << pulse_confidence << "), "
                  << "Breathing: " << breathing_rate << " BPM (" << breathing_confidence << ")";

        return absl::OkStatus();
    }
));

// Edge metrics callback (real-time on-device processing)
MP_RETURN_IF_ERROR(container->SetOnEdgeMetricsOutput(
    [](const presage::physiology::Metrics& metrics) {
        // Access real-time breathing trace
        const auto& upper_trace = metrics.breathing().upper_trace();
        if (!upper_trace.empty()) {
            const auto& latest_sample = *upper_trace.rbegin();
            float time = latest_sample.time();
            float value = latest_sample.value();
            bool stable = latest_sample.stable();

            // Process real-time breathing data
        }

        return absl::OkStatus();
    }
));

// Video output callback (for GUI/display)
MP_RETURN_IF_ERROR(container->SetOnVideoOutput(
    [](cv::Mat& output_frame, int64_t timestamp_milliseconds) {
        // Draw custom overlays
        cv::putText(output_frame, "SmartSpectra Active",
                   cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                   cv::Scalar(0, 255, 0), 2);

        // Display frame
        cv::imshow("SmartSpectra", output_frame);
        cv::waitKey(1);

        return absl::OkStatus();
    }
));

// Status change callback
MP_RETURN_IF_ERROR(container->SetOnStatusChange(
    [](presage::physiology::StatusValue status) {
        presage::physiology::StatusCode status_code = status.value();
        std::string status_description = presage::physiology::GetStatusDescription(status_code);
        LOG(INFO) << "Status: " << status_description;

        // Handle specific statuses
        switch (status_code) {
            case presage::physiology::StatusCode::kFaceNotVisible:
                // Guide user to position face properly
                break;
            case presage::physiology::StatusCode::kTooMuchMovement:
                // Ask user to remain still
                break;
            case presage::physiology::StatusCode::kPoorLighting:
                // Suggest better lighting
                break;
            default:
                break;
        }

        return absl::OkStatus();
    }
));
```

### GUI Components

```cpp
// HUD for displaying metrics overlay
presage::smartspectra::gui::OpenCvHud hud(10, 0, 1260, 400);

// Update HUD with new metrics
hud.UpdateWithNewMetrics(metrics_buffer);

// Render HUD on frame
auto status = hud.Render(output_frame);

// Trace plotter for real-time signals
presage::smartspectra::gui::OpenCvTracePlotter plotter(10, 450, 910, 100);

// Update with edge metrics
const auto& upper_trace = edge_metrics.breathing().upper_trace();
if (!upper_trace.empty()) {
    plotter.UpdateTraceWithSample(*upper_trace.rbegin());
}

// Render plotter
auto status = plotter.Render(output_frame, cv::Scalar(0, 165, 255));

// Value indicators for performance metrics
presage::smartspectra::gui::OpenCvValueIndicator fps_indicator(1200, 580, 60, 60);
presage::smartspectra::gui::OpenCvLabel fps_label(920, 565, 270, 60, "FPS:");

// Update and render
fps_indicator.Render(output_frame, current_fps, cv::Scalar(40, 200, 0));
fps_label.Render(output_frame, cv::Scalar(40, 200, 0));
```

### Data Structures

```cpp
// MetricsBuffer (main server response)
presage::physiology::MetricsBuffer metrics_buffer;

// Pulse data access
const auto& pulse = metrics_buffer.pulse();
float strict_pulse_rate = pulse.strict().value();  // Final pulse rate

// Rate measurements over time
for (int i = 0; i < pulse.rate_size(); ++i) {
    const auto& rate_measurement = pulse.rate(i);
    float time = rate_measurement.time();
    float value = rate_measurement.value();
    float confidence = rate_measurement.confidence();
}

// Raw pulse trace
for (int i = 0; i < pulse.trace_size(); ++i) {
    const auto& trace_point = pulse.trace(i);
    float time = trace_point.time();
    float value = trace_point.value();
}

// Breathing data access
const auto& breathing = metrics_buffer.breathing();
float strict_breathing_rate = breathing.strict().value();

// Breathing trace (upper)
for (const auto& trace_point : breathing.upper_trace()) {
    float time = trace_point.time();
    float value = trace_point.value();
}

// Breathing amplitude
for (const auto& amplitude_point : breathing.amplitude()) {
    float time = amplitude_point.time();
    float value = amplitude_point.value();
}

// Apnea detection
for (const auto& apnea_event : breathing.apnea()) {
    float time = apnea_event.time();
    bool detected = apnea_event.detected();
}

// Blood pressure (if enabled)
if (metrics_buffer.has_blood_pressure()) {
    const auto& bp = metrics_buffer.blood_pressure();
    for (const auto& phasic_point : bp.phasic()) {
        float time = phasic_point.time();
        float value = phasic_point.value();
    }
}

// Face metrics
const auto& face = metrics_buffer.face();
for (const auto& blink_event : face.blinking()) {
    float time = blink_event.time();
    bool detected = blink_event.detected();
}

for (const auto& talk_event : face.talking()) {
    float time = talk_event.time();
    bool detected = talk_event.detected();
}

// Metadata
if (metrics_buffer.has_metadata()) {
    const auto& metadata = metrics_buffer.metadata();
    std::string id = metadata.id();
    int64_t upload_timestamp = metadata.upload_timestamp();
}
```

## Integration Patterns

### Thread-Safe Metrics Access

```cpp
// Thread-safe metrics storage
class MetricsStorage {
public:
    void UpdateMetrics(const presage::physiology::MetricsBuffer& metrics) {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_metrics_ = metrics;
        metrics_history_.push_back({std::chrono::steady_clock::now(), metrics});

        // Limit history size
        if (metrics_history_.size() > max_history_size_) {
            metrics_history_.pop_front();
        }
    }

    std::optional<presage::physiology::MetricsBuffer> GetLatestMetrics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return latest_metrics_;
    }

    std::vector<TimestampedMetrics> GetMetricsHistory(std::chrono::seconds window) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto cutoff = std::chrono::steady_clock::now() - window;

        std::vector<TimestampedMetrics> result;
        for (const auto& entry : metrics_history_) {
            if (entry.timestamp >= cutoff) {
                result.push_back(entry);
            }
        }
        return result;
    }

private:
    struct TimestampedMetrics {
        std::chrono::steady_clock::time_point timestamp;
        presage::physiology::MetricsBuffer metrics;
    };

    mutable std::mutex mutex_;
    std::optional<presage::physiology::MetricsBuffer> latest_metrics_;
    std::deque<TimestampedMetrics> metrics_history_;
    size_t max_history_size_ = 1000;
};
```

### Asynchronous Processing

```cpp
// Asynchronous measurement processing
class AsyncMeasurementProcessor {
public:
    AsyncMeasurementProcessor() : worker_thread_(&AsyncMeasurementProcessor::ProcessingLoop, this) {}

    ~AsyncMeasurementProcessor() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            should_stop_ = true;
        }
        queue_cv_.notify_all();
        worker_thread_.join();
    }

    void QueueMetrics(const presage::physiology::MetricsBuffer& metrics) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            metrics_queue_.push(metrics);
        }
        queue_cv_.notify_one();
    }

    void SetResultCallback(std::function<void(const ProcessingResult&)> callback) {
        result_callback_ = std::move(callback);
    }

private:
    struct ProcessingResult {
        std::string analysis;
        double confidence;
        std::chrono::steady_clock::time_point timestamp;
    };

    void ProcessingLoop() {
        while (true) {
            presage::physiology::MetricsBuffer metrics;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] { return !metrics_queue_.empty() || should_stop_; });

                if (should_stop_) break;

                metrics = metrics_queue_.front();
                metrics_queue_.pop();
            }

            // Perform heavy processing
            auto result = ProcessMetrics(metrics);

            if (result_callback_) {
                result_callback_(result);
            }
        }
    }

    ProcessingResult ProcessMetrics(const presage::physiology::MetricsBuffer& metrics) {
        // Complex analysis implementation
        ProcessingResult result;
        result.timestamp = std::chrono::steady_clock::now();

        // Analyze pulse patterns
        float pulse_rate = metrics.pulse().strict().value();
        if (pulse_rate > 0) {
            if (pulse_rate < 60) {
                result.analysis = "Bradycardia detected";
                result.confidence = 0.8;
            } else if (pulse_rate > 100) {
                result.analysis = "Tachycardia detected";
                result.confidence = 0.8;
            } else {
                result.analysis = "Normal pulse rate";
                result.confidence = 0.9;
            }
        }

        return result;
    }

    std::thread worker_thread_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<presage::physiology::MetricsBuffer> metrics_queue_;
    bool should_stop_ = false;
    std::function<void(const ProcessingResult&)> result_callback_;
};
```

### Data Export and Persistence

```cpp
// Data export utilities
class DataExporter {
public:
    // Export to CSV format
    static absl::Status ExportToCSV(
        const std::vector<presage::physiology::MetricsBuffer>& metrics_list,
        const std::string& filepath) {

        std::ofstream file(filepath);
        if (!file.is_open()) {
            return absl::InvalidArgumentError("Cannot open file: " + filepath);
        }

        // Write header
        file << "timestamp,pulse_rate,pulse_confidence,breathing_rate,breathing_confidence\n";

        // Write data
        for (const auto& metrics : metrics_list) {
            float pulse_rate = metrics.pulse().strict().value();
            float breathing_rate = metrics.breathing().strict().value();

            float pulse_confidence = 0.0f;
            if (metrics.pulse().rate_size() > 0) {
                pulse_confidence = metrics.pulse().rate(metrics.pulse().rate_size() - 1).confidence();
            }

            float breathing_confidence = 0.0f;
            if (metrics.breathing().rate_size() > 0) {
                breathing_confidence = metrics.breathing().rate(metrics.breathing().rate_size() - 1).confidence();
            }

            auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            file << timestamp << "," << pulse_rate << "," << pulse_confidence << ","
                 << breathing_rate << "," << breathing_confidence << "\n";
        }

        return absl::OkStatus();
    }

    // Export to JSON format
    static absl::Status ExportToJSON(
        const std::vector<presage::physiology::MetricsBuffer>& metrics_list,
        const std::string& filepath) {

        std::ofstream file(filepath);
        if (!file.is_open()) {
            return absl::InvalidArgumentError("Cannot open file: " + filepath);
        }

        file << "[\n";
        for (size_t i = 0; i < metrics_list.size(); ++i) {
            std::string metrics_json;
            google::protobuf::util::JsonPrintOptions options;
            options.add_whitespace = true;

            auto status = google::protobuf::util::MessageToJsonString(
                metrics_list[i], &metrics_json, options);
            if (!status.ok()) {
                return absl::InternalError("Failed to convert to JSON: " + status.message());
            }

            file << "  " << metrics_json;
            if (i < metrics_list.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        file << "]\n";

        return absl::OkStatus();
    }

    // Export time series data
    static absl::Status ExportTimeSeriesCSV(
        const presage::physiology::MetricsBuffer& metrics,
        const std::string& filepath,
        const std::string& signal_type = "pulse") {

        std::ofstream file(filepath);
        if (!file.is_open()) {
            return absl::InvalidArgumentError("Cannot open file: " + filepath);
        }

        file << "time,value\n";

        if (signal_type == "pulse") {
            for (const auto& point : metrics.pulse().trace()) {
                file << point.time() << "," << point.value() << "\n";
            }
        } else if (signal_type == "breathing") {
            for (const auto& point : metrics.breathing().upper_trace()) {
                file << point.time() << "," << point.value() << "\n";
            }
        }

        return absl::OkStatus();
    }
};
```

## Build Configuration

### CMakeLists.txt - Complete Build Setup

```cmake
# CMakeLists.txt - Complete build configuration
cmake_minimum_required(VERSION 3.27.0)
project(SmartSpectraApp CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(SmartSpectra REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)
find_package(glog REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Define executable targets
add_executable(continuous_app
    continuous_app.cc
    SmartSpectraProcessor.cpp
    SmartSpectraGUI.cpp
    MetricsAnalyzer.cpp
)

add_executable(spot_app
    spot_app.cc
    SmartSpectraProcessor.cpp
)

add_executable(minimal_example
    minimal_example.cc
)

# Link libraries
target_link_libraries(continuous_app
    SmartSpectra::Container
    SmartSpectra::Gui
    ${OpenCV_LIBS}
    protobuf::libprotobuf
    absl::status
    absl::flags
    absl::flags_parse
    glog::glog
)

target_link_libraries(spot_app
    SmartSpectra::Container
    SmartSpectra::Gui
    ${OpenCV_LIBS}
    protobuf::libprotobuf
    absl::status
    absl::flags
    absl::flags_parse
    glog::glog
)

target_link_libraries(minimal_example
    SmartSpectra::Container
    ${OpenCV_LIBS}
    glog::glog
)

# Compiler flags
target_compile_options(continuous_app PRIVATE
    -Wall -Wextra -O2
)

target_compile_options(spot_app PRIVATE
    -Wall -Wextra -O2
)

target_compile_options(minimal_example PRIVATE
    -Wall -Wextra -O2
)

# Install targets
install(TARGETS continuous_app spot_app minimal_example
    RUNTIME DESTINATION bin
)

# Custom build configurations
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(continuous_app PRIVATE DEBUG_BUILD)
    target_compile_definitions(spot_app PRIVATE DEBUG_BUILD)
endif()

# Package configuration
set(CPACK_PACKAGE_NAME "SmartSpectraApp")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION "SmartSpectra C++ Application")
set(CPACK_GENERATOR "DEB;TGZ")
include(CPack)
```

### Build Scripts

```bash
#!/bin/bash
# build.sh - Build script for SmartSpectra applications

set -e

# Configuration
BUILD_TYPE=${1:-Release}
BUILD_DIR="build"
INSTALL_PREFIX="/usr/local"

echo "Building SmartSpectra applications..."
echo "Build type: $BUILD_TYPE"

# Create build directory
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build
make -j$(nproc)

# Run tests if available
if [ -f CTestTestfile.cmake ]; then
    echo "Running tests..."
    ctest --output-on-failure
fi

echo "Build completed successfully!"
echo "Executables are in: $BUILD_DIR/"
echo "To install system-wide, run: sudo make install"
```

## Code Snippets Library

### Common Patterns

```cpp
// Pattern: Safe metric access with validation
class SafeMetricsAccessor {
public:
    static std::optional<int> GetPulseRate(const presage::physiology::MetricsBuffer& metrics) {
        if (!metrics.has_pulse() || metrics.pulse().strict().value() <= 0) {
            return std::nullopt;
        }
        return static_cast<int>(metrics.pulse().strict().value());
    }

    static std::optional<int> GetBreathingRate(const presage::physiology::MetricsBuffer& metrics) {
        if (!metrics.has_breathing() || metrics.breathing().strict().value() <= 0) {
            return std::nullopt;
        }
        return static_cast<int>(metrics.breathing().strict().value());
    }

    static bool IsValidMeasurement(const presage::physiology::MetricsBuffer& metrics) {
        auto pulse = GetPulseRate(metrics);
        auto breathing = GetBreathingRate(metrics);
        return pulse.has_value() && breathing.has_value() &&
               *pulse > 0 && *breathing > 0;
    }
};

// Pattern: RAII container management
class ContainerManager {
public:
    template<typename ContainerType, typename SettingsType>
    static std::unique_ptr<ContainerType> CreateAndInitialize(const SettingsType& settings) {
        auto container = std::make_unique<ContainerType>(settings);

        auto status = container->Initialize();
        if (!status.ok()) {
            LOG(ERROR) << "Failed to initialize container: " << status.message();
            return nullptr;
        }

        return container;
    }
};

// Pattern: Error handling with status chaining
#define RETURN_IF_ERROR(expr) \
    do { \
        const auto& status = (expr); \
        if (!status.ok()) { \
            return status; \
        } \
    } while (0)

absl::Status SetupProcessingPipeline(auto& container) {
    RETURN_IF_ERROR(container->SetOnCoreMetricsOutput(metrics_callback));
    RETURN_IF_ERROR(container->SetOnVideoOutput(video_callback));
    RETURN_IF_ERROR(container->SetOnStatusChange(status_callback));
    RETURN_IF_ERROR(container->Initialize());
    
    return absl::OkStatus();
}
```

### Performance Optimization

```cpp
// Performance monitoring utilities
class PerformanceMonitor {
public:
    class Timer {
    public:
        Timer(const std::string& name) : name_(name), start_(std::chrono::high_resolution_clock::now()) {}
        
        ~Timer() {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
            LOG(INFO) << name_ << " took: " << duration.count() << " μs";
        }
        
    private:
        std::string name_;
        std::chrono::high_resolution_clock::time_point start_;
    };
    
    static void TrackMetricsProcessing(const presage::physiology::MetricsBuffer& metrics) {
        Timer timer("Metrics Processing");
        
        // Track processing time for different metric types
        if (metrics.has_pulse()) {
            Timer pulse_timer("Pulse Processing");
            // Process pulse data
        }
        
        if (metrics.has_breathing()) {
            Timer breathing_timer("Breathing Processing");
            // Process breathing data
        }
    }
};

// Memory-efficient metrics storage
class CompactMetricsStorage {
public:
    struct CompactMetrics {
        std::chrono::steady_clock::time_point timestamp;
        float pulse_rate;
        float breathing_rate;
        float pulse_confidence;
        float breathing_confidence;
    };
    
    void AddMetrics(const presage::physiology::MetricsBuffer& metrics) {
        CompactMetrics compact;
        compact.timestamp = std::chrono::steady_clock::now();
        compact.pulse_rate = metrics.pulse().strict().value();
        compact.breathing_rate = metrics.breathing().strict().value();
        
        if (metrics.pulse().rate_size() > 0) {
            compact.pulse_confidence = metrics.pulse().rate(metrics.pulse().rate_size() - 1).confidence();
        }
        
        if (metrics.breathing().rate_size() > 0) {
            compact.breathing_confidence = metrics.breathing().rate(metrics.breathing().rate_size() - 1).confidence();
        }
        
        storage_.push_back(compact);
        
        // Limit storage size for memory efficiency
        if (storage_.size() > max_entries_) {
            storage_.erase(storage_.begin());
        }
    }
    
    const std::vector<CompactMetrics>& GetStorage() const { return storage_; }

private:
    std::vector<CompactMetrics> storage_;
    size_t max_entries_ = 10000;
};
```

### Error Handling and Recovery

```cpp
// Robust error handling and recovery
class RobustProcessor {
public:
    absl::Status RunWithRetry(int max_attempts = 3) {
        for (int attempt = 1; attempt <= max_attempts; ++attempt) {
            LOG(INFO) << "Attempt " << attempt << " of " << max_attempts;
            
            auto status = TryRun();
            if (status.ok()) {
                return absl::OkStatus();
            }
            
            LOG(WARNING) << "Attempt " << attempt << " failed: " << status.message();
            
            if (attempt < max_attempts) {
                // Exponential backoff
                auto delay = std::chrono::seconds(static_cast<int>(std::pow(2, attempt - 1)));
                LOG(INFO) << "Waiting " << delay.count() << " seconds before retry...";
                std::this_thread::sleep_for(delay);
            }
        }
        
        return absl::DeadlineExceededError("Max retry attempts exceeded");
    }

private:
    absl::Status TryRun() {
        try {
            // Initialize processor
            auto status = processor_->Initialize();
            if (!status.ok()) {
                return status;
            }
            
            // Run with timeout
            auto future = std::async(std::launch::async, [this]() {
                return processor_->Start();
            });
            
            auto timeout = std::chrono::seconds(30);
            if (future.wait_for(timeout) == std::future_status::timeout) {
                return absl::DeadlineExceededError("Operation timed out");
            }
            
            return future.get();
            
        } catch (const std::exception& e) {
            return absl::InternalError("Exception caught: " + std::string(e.what()));
        }
    }
    
    std::unique_ptr<my_app::SmartSpectraProcessor> processor_;
};
```

## C++ Requirements and Notes

- **Minimum C++ Standard**: C++17
- **Supported Systems**: Ubuntu 22.04/Mint 21 (x86_64), other platforms via partners
- **Required Dependencies**: OpenCV, glog, abseil, protobuf, MediaPipe
- **Camera Requirements**: USB/V4L2 compatible camera
- **Network**: Required for REST API authentication and cloud processing
- **Authentication**: API Key or OnPrem gRPC server
- **Threading**: Thread-safe design with proper synchronization

## Common Customization Points

1. **Container Selection**: Choose between Spot/Continuous and REST/gRPC modes
2. **Video Source**: Camera, video file, or custom input streams
3. **GUI Components**: OpenCV-based HUD, plotters, and indicators
4. **Data Processing**: Custom callbacks for metrics analysis
5. **Performance Tuning**: Buffer sizes, frame rates, and processing parameters
6. **Data Export**: JSON, CSV, or custom format exporters

## Integration Checklist

- [ ] Install libsmartspectra-dev package or build from source
- [ ] Set up CMakeLists.txt with proper package finding
- [ ] Configure authentication (API key or gRPC server)
- [ ] Choose appropriate container type for your use case
- [ ] Implement required callbacks for metrics and video processing
- [ ] Set up error handling and status monitoring
- [ ] Test with physical camera (simulators not supported)
- [ ] Implement data persistence if needed
- [ ] Configure GUI components as desired
- [ ] Add performance monitoring and optimization

## Typical Parameter Values

- **Spot Duration**: 30 seconds (default), 20-120 seconds (range)
- **Buffer Duration**: 0.5 seconds (continuous mode)
- **Camera Resolution**: 1280x720 (default), configurable
- **Frame Rate**: 30 FPS typical
- **Interframe Delay**: 20ms (default)
- **Verbosity Level**: 1 (default), 0-3 (range)
