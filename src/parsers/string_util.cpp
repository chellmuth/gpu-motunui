#include "moana/parsers/string_util.hpp"

namespace moana { namespace StringUtil {

std::queue<std::string> tokenize(const std::string &line)
{
    std::queue<std::string> tokens;
    std::string remaining = lTrim(line);

    while(remaining.length() > 0) {
        std::string::size_type endContentIndex = remaining.find_first_of(" \t");
        if (endContentIndex == std::string::npos) {
            tokens.push(remaining);
            return tokens;
        }

        tokens.push(remaining.substr(0, endContentIndex));
        remaining = lTrim(remaining.substr(endContentIndex));
    }

    return tokens;
}

std::string lTrim(const std::string &token)
{
    std::string::size_type firstContentIndex = token.find_first_not_of(" \t");
    if (firstContentIndex == 0) {
        return std::string(token);
    } else if (firstContentIndex == std::string::npos) {
        return "";
    }

    return token.substr(firstContentIndex);
}

} }
