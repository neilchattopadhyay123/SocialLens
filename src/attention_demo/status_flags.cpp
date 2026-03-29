#include "attention_demo/status_flags.hpp"

#include <algorithm>
#include <cctype>

namespace attention_demo {

StatusFlags ParseStatusFlagsFromDescription(const std::string& description) {
    std::string lower = description;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    StatusFlags flags;
    flags.face_not_visible =
        lower.find("face") != std::string::npos &&
        lower.find("visible") != std::string::npos;
    flags.too_much_movement =
        lower.find("movement") != std::string::npos ||
        lower.find("motion") != std::string::npos;
    flags.poor_lighting = lower.find("light") != std::string::npos;

    return flags;
}

}  // namespace attention_demo
