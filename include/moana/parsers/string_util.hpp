#pragma once

#include <string>
#include <queue>

namespace moana { namespace StringUtil {

std::queue<std::string> tokenize(const std::string &line);
std::string lTrim(const std::string &token);

} }
