#pragma once

#include <chrono>
#include <map>
#include <mutex>


namespace moana {

enum class TimedSection {
    Sample,
    PtexLookups,
    DirectLighting
};

class Timing {
public:
    Timing();

    void start(TimedSection section);
    void end(TimedSection section);
    float getMilliseconds(TimedSection section);

private:
    std::map<TimedSection, std::chrono::steady_clock::time_point> m_startTimes;
    std::map<TimedSection, std::chrono::steady_clock::duration> m_durations;

    std::mutex m_lock;
};

}
