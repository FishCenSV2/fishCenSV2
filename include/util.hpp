#include <chrono>

namespace util {

    //Implementation of std::accumulate Found from the cppreference site.
    template<typename InputIt, typename T>
    constexpr
    T accumulate(InputIt first, InputIt last, T init) {
        for(; first != last; ++first) {
            init = std::move(init) + *first;
        }

        return init;
    } 

    class StopWatch {
        private:
            std::chrono::high_resolution_clock::time_point _start_time;
            std::chrono::high_resolution_clock::time_point _end_time;
        public:
            using seconds = std::chrono::seconds;
            using milliseconds = std::chrono::milliseconds;
            using microseconds = std::chrono::microseconds;
            using nanoseconds = std::chrono::nanoseconds;

            inline void start() noexcept {
                _start_time = std::chrono::high_resolution_clock::now();
            };
            inline void end() noexcept {
                _end_time = std::chrono::high_resolution_clock::now();
            };
            template<typename T>
            inline long long get_time() {
                return std::chrono::duration_cast<T>(_end_time-_start_time).count();
            };
    };
}