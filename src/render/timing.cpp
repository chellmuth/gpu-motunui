#include "render/timing.hpp"

namespace moana {

void Timing::start(TimedSection section)
{
    m_startTimes[section] = std::chrono::steady_clock::now();
}

void Timing::end(TimedSection section)
{
    m_durations[section] = std::chrono::steady_clock::now() - m_startTimes[section];
}

float Timing::getMilliseconds(TimedSection section)
{
    using namespace std::chrono;
    const auto duration = duration_cast<milliseconds>(m_durations[section]);
    return duration.count() / 1000.f;
}

}
