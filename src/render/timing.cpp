#include "render/timing.hpp"

namespace moana {

Timing::Timing()
    : m_startTimes(),
      m_durations(),
      m_lock()
{}

void Timing::start(TimedSection section)
{
    m_lock.lock();
    m_startTimes[section] = std::chrono::steady_clock::now();
    m_lock.unlock();
}

void Timing::end(TimedSection section)
{
    m_lock.lock();
    m_durations[section] = std::chrono::steady_clock::now() - m_startTimes[section];
    m_lock.unlock();
}

float Timing::getMilliseconds(TimedSection section)
{
    using namespace std::chrono;

    m_lock.lock();
    const auto duration = duration_cast<milliseconds>(m_durations[section]);
    m_lock.unlock();

    return duration.count() / 1000.f;
}

}
