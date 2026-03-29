#pragma once

#include <string>

namespace attention_demo {

class DemoVideoApp {
public:
    int Run(const std::string& api_key,
            const std::string& input_video_path,
            const std::string& output_video_path);
};

}  // namespace attention_demo
