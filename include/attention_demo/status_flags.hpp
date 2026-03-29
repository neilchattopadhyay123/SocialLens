#pragma once

#include <string>

namespace attention_demo {

struct StatusFlags {
    bool face_not_visible = false;
    bool too_much_movement = false;
    bool poor_lighting = false;
};

StatusFlags ParseStatusFlagsFromDescription(const std::string& description);

}  // namespace attention_demo
