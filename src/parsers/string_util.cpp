#include "moana/parsers/string_util.hpp"

namespace moana { namespace StringUtil {

std::optional<std::string_view> lTrim(const std::string_view &token)
{
    std::string::size_type firstContentIndex = token.find_first_not_of(" \t");
    if (firstContentIndex == 0) {
        return token;
    } else if (firstContentIndex == std::string::npos) {
        return std::nullopt;
    }

    return token.substr(firstContentIndex);
}

} }
