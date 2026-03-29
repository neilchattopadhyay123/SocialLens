#include "attention_demo/demo_video_app.hpp"

#include <glog/logging.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_alsologtostderr = true;

    std::string api_key;
    if (argc > 1) {
        api_key = argv[1];
    } else if (const char* env_key = std::getenv("SMARTSPECTRA_API_KEY")) {
        api_key = env_key;
    } else {
        std::cout << "Usage: ./create_demo_video YOUR_API_KEY [input_video_path] [output_video_path]\n";
        std::cout << "Or set SMARTSPECTRA_API_KEY environment variable\n";
        return 1;
    }

    const std::string input_video_path = (argc > 2) ? argv[2] : "initial_test/test.mp4";
    const std::string output_video_path = (argc > 3) ? argv[3] : "attention_demo.mp4";

    attention_demo::DemoVideoApp app;
    return app.Run(api_key, input_video_path, output_video_path);
}
