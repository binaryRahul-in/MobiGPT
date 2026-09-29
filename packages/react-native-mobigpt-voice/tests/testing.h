// Minimal dependency-free test harness (keeps the host build offline-friendly).
#pragma once

#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mt {

struct Case {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}

struct Registrar {
  Registrar(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};

struct Failure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::string fixture(const std::string& name) {
#ifdef MOBIGPT_FIXTURES
  return std::string(MOBIGPT_FIXTURES) + "/" + name;
#else
  return "fixtures/" + name;
#endif
}

inline bool fileExists(const std::string& p) {
  FILE* f = std::fopen(p.c_str(), "rb");
  if (f) std::fclose(f);
  return f != nullptr;
}

}  // namespace mt

#define MT_CAT2(a, b) a##b
#define MT_CAT(a, b) MT_CAT2(a, b)
#define TEST(name)                                                   \
  static void name();                                                \
  static mt::Registrar MT_CAT(reg_, name)(#name, name);              \
  static void name()

#define REQUIRE(cond)                                                                        \
  do {                                                                                       \
    if (!(cond)) {                                                                           \
      std::ostringstream os_;                                                                \
      os_ << __FILE__ << ":" << __LINE__ << ": REQUIRE(" #cond ") failed";                   \
      throw mt::Failure(os_.str());                                                          \
    }                                                                                        \
  } while (0)

#define REQUIRE_NEAR(a, b, tol)                                                              \
  do {                                                                                       \
    const double a_ = (a), b_ = (b);                                                         \
    if (!(std::fabs(a_ - b_) <= (tol))) {                                                    \
      std::ostringstream os_;                                                                \
      os_ << __FILE__ << ":" << __LINE__ << ": " #a " = " << a_ << ", " #b " = " << b_       \
          << " (tol " << (tol) << ")";                                                       \
      throw mt::Failure(os_.str());                                                          \
    }                                                                                        \
  } while (0)

#define REQUIRE_THROWS(expr)                                                                 \
  do {                                                                                       \
    bool threw_ = false;                                                                     \
    try {                                                                                    \
      expr;                                                                                  \
    } catch (...) {                                                                          \
      threw_ = true;                                                                         \
    }                                                                                        \
    if (!threw_) throw mt::Failure(std::string(__FILE__) + ": expected exception: " #expr); \
  } while (0)

#define SKIP_UNLESS_FIXTURE(name)                                                            \
  do {                                                                                       \
    if (!mt::fileExists(mt::fixture(name))) {                                                \
      std::printf("    (skipped: fixture %s missing)\n", name);                             \
      return;                                                                                \
    }                                                                                        \
  } while (0)
