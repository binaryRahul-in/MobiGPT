#include <chrono>
#include <cstdio>
#include <cstring>

#include "testing.h"

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int passed = 0, failed = 0;
  for (const auto& c : mt::registry()) {
    if (filter && !std::strstr(c.name, filter)) continue;
    const auto t0 = std::chrono::steady_clock::now();
    try {
      c.fn();
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::printf("[PASS] %s (%.0f ms)\n", c.name, ms);
      ++passed;
    } catch (const std::exception& e) {
      std::printf("[FAIL] %s\n       %s\n", c.name, e.what());
      ++failed;
    }
  }
  std::printf("\n%d passed, %d failed\n", passed, failed);
  return failed == 0 ? 0 : 1;
}
