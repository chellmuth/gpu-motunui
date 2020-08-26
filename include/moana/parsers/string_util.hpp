#pragma once

#include <optional>
#include <string_view>

namespace moana { namespace StringUtil {

std::optional<std::string_view> lTrim(const std::string_view &token);

} }
