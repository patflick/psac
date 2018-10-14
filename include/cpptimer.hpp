#ifndef CPPTIMER_H
#define CPPTIMER_H

#include <iostream>
#include <chrono>
#include <unistd.h> // gethostname
#include <limits.h> // HOST_NAME_MAX

#include <fstream>

struct timed_section {

    using clock_type = std::chrono::steady_clock;
    using time_point = clock_type::time_point;
    using precision = std::chrono::microseconds;

    time_point ts;
    std::string name;
    int mylevel;

    timed_section(const std::string& name) : name(name) {
        mylevel = level();
        level()++;
        start();
    }

    inline void print_prefix() {
        std::cout << "[TIMER] ";
        for (int i = 0; i < mylevel; ++i) {
          std::cout << "    ";
        }
    }

    inline void start() {
        ts = clock_type::now();
    }

    inline void print_time(const time_point::duration& dur) {
        print_prefix();
        auto d = std::chrono::duration_cast<precision>(dur).count();
        std::cout << name << ": " << d << " μs" << std::endl;
    }

    inline void new_section(const std::string& name) {
        // print out previous section, then start new
        auto ets = clock_type::now();
        print_time(ets - ts);
        this->name = name;
        start();
    }

    inline void end() {
        auto ets = clock_type::now();
        print_time(ets - ts);
    }

    static int& level() {
      static int level = 0;
      return level;
    }

    virtual ~timed_section() {
        end();
        level()--;
    }
};

inline std::string get_hostname() {
  char hostname[HOST_NAME_MAX];
  gethostname(hostname, HOST_NAME_MAX);
  return std::string(hostname);
}


struct timer {
    using clock_type = std::chrono::steady_clock;
    using time_point = clock_type::time_point;
    using duration = time_point::duration;

    time_point ts;
    duration elapsed;
    duration total;

    inline void tic() {
      ts = clock_type::now();
    }

    inline void toc() {
      elapsed = clock_type::now() - ts;
      total += elapsed;
    }

    template <typename precision>
    inline typename precision::rep get_time() const {
      return std::chrono::duration_cast<precision>(elapsed).count();
    }

    inline std::chrono::nanoseconds::rep get_ns() const {
      return get_time<std::chrono::nanoseconds>();
    }
};


struct benchmark_timer {

  // execution context
  std::chrono::system_clock::time_point start_time;
  std::string hostname; // log the machine this is run on
  int nthreads;

  // log file
  std::ofstream os;

  // benchmark run info
  std::string tensor;
  std::string method;
  std::string version;
  int mode;

  // timer implementing tic toc
  timer t;

  benchmark_timer(const std::string& logfile) : os(logfile) {
    setup_context();
  }

  // setup execution info
  void setup_context() {
    start_time = std::chrono::system_clock::now();
    hostname = get_hostname();
    nthreads = omp_get_max_threads();
  }

  // setup benchmark run info
  void set_method(const std::string& method, const std::string& version = "") {
    this->method = method;
    this->version = version;
  }
  void set_mode(int mode) {
    this->mode = mode;
  }
  void set_tensor(const std::string& tensor) {
    this->tensor = tensor;
  }

  void log_time(long long t) {
    // specialized output?
    char sep = ';';
    os << hostname << sep << nthreads << sep << tensor << sep << method << sep << version << sep << mode << sep << t << std::endl;
  }

  inline void tic() {
    t.tic();
  }
  inline void toc() {
    t.toc();
    log_time(t.get_ns());
  }
};

#endif // CPPTIMER_H
