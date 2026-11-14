#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <stdint.h>

#ifdef TIMER

#define TIMER_CREATE(name) \
    struct timespec _timer_##name##_start, \
                    _timer_##name##_end; \
    static double _timer_##name##_elapsed_ms = 0; \
    static uint64_t _timer_##name##_count = 0;

#define TIMER_RESTART(name) \
do { \
    clock_gettime(CLOCK_MONOTONIC, &_timer_##name##_start); \
} while (0)

#define TIMER_STOP(name) \
do { \
    clock_gettime(CLOCK_MONOTONIC, &_timer_##name##_end); \
    _timer_##name##_elapsed_ms += (_timer_##name##_end.tv_sec - \
                                   _timer_##name##_start.tv_sec) * 1000 + \
                                  (_timer_##name##_end.tv_nsec - \
                                   _timer_##name##_start.tv_nsec) / 1e6; \
    ++_timer_##name##_count; \
} while (0)

#define TIMER_LOG_IF(name, cond) \
if (cond) { \
    printf("%-16s %12.4f ms %12s   [avg %ld runs]\n", #name, \
           _timer_##name##_elapsed_ms / _timer_##name##_count, "", \
           _timer_##name##_count); \
}

#define TIMER_LOG_BW_IF(name, bytes, cond) \
if (cond) { \
    double bandwidth = (bytes) / \
        (_timer_##name##_elapsed_ms / _timer_##name##_count) / 1e6; \
    printf("%-16s %12.4f ms %7.2f GB/s   [avg %ld runs]\n", #name, \
           _timer_##name##_elapsed_ms / _timer_##name##_count, \
           bandwidth, _timer_##name##_count); \
}

#define TIMER_LOG_NL_IF(cond) \
if (cond) { \
    printf("\n"); \
}

#define TIMER_LOG(name) TIMER_LOG_IF(name, 1)
#define TIMER_LOG_BW(name, bytes) TIMER_LOG_BW_IF(name, bytes, 1)
#define TIMER_LOG_NL() TIMER_LOG_NL_IF(1)

#else

#define TIMER_CREATE(...)
#define TIMER_RESTART(...)
#define TIMER_STOP(...)
#define TIMER_LOG_IF(...)
#define TIMER_LOG_BW_IF(...)
#define TIMER_LOG_NL_IF(...)
#define TIMER_LOG(...)
#define TIMER_LOG_BW(...)
#define TIMER_LOG_NL(...)

#endif

#endif
