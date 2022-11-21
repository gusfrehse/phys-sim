#pragma once

// disclaimer: almost entirely based on The Cherno's video on visual profiling.

#include <bits/chrono.h>
#include <chrono>
#include <cstdio>
#include <vector>


#ifdef PROFILE

#define PROFILE_FUNC() scope_timer timer##__LINE__(__FUNCTION__)

#else

#define PROFILE_FUNC() 

#endif

struct profile_info {
  const char* name;
  long long start;
  long long duration;
};

class profiler {
private:
  std::vector<profile_info> m_infos;
  long long m_samples = 0;
  FILE* m_out_file;
  const char *m_file_path;

  profiler(const char* file_path = "profile.json") {
    m_file_path = file_path;
    m_out_file = fopen(file_path, "w");
    fprintf(m_out_file, "[\n");
    fflush(m_out_file);
  }

  ~profiler() {
    fprintf(stderr,
            "profiler made %lld samples and saved at '%s'\n",
            m_samples,
            m_file_path);
    fprintf(m_out_file, "]\n");
    fclose(m_out_file);
  }

public:
  static profiler& get_instance() {
    static profiler p;
    return p;
  }

  void record(profile_info info) {
    if (m_samples++)
      fprintf(m_out_file, ",");

    fprintf(m_out_file,
            "  {\n"
            "    \"name\":\"%s\","
            "    \"cat\" :\"function\","
            "    \"ph\"  :\"X\","
            "    \"ts\"  :\"%lld\","
            "    \"dur\" :\"%lld\","
            "    \"pid\" :\"0\","
            "    \"tid\" :\"0\"\n"
            "  }\n",
            info.name,
            info.start,
            info.duration);

    fflush(m_out_file);
  }

};

using timer_clock = std::chrono::steady_clock;

class scope_timer {
private:
  timer_clock::time_point m_start;
  const char *m_name;

public:
  scope_timer(const char *name) : 
    m_start(timer_clock::now()), m_name(name) {}

  ~scope_timer() {
    long long now =
      std::chrono::time_point_cast<std::chrono::microseconds>(timer_clock::now())
      .time_since_epoch().count();

    long long start =
      std::chrono::time_point_cast<std::chrono::microseconds>(m_start)
      .time_since_epoch().count();

    profiler::get_instance().record({ m_name, start, now - start });
  }
};

/* vim: set sts=2 ts=2 sw=2 et cc=81: */

