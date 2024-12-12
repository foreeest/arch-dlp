#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <immintrin.h>

using std::vector;

class FigureProcessor {
private:
  unsigned char* figure;
  unsigned char* result;
  const size_t size;

public:
  FigureProcessor(size_t size, size_t seed = 0) : size(size) {
    figure = new unsigned char[size * size];
    result = new unsigned char[size * size];

    // !!! Please do not modify the following code !!!
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // 初始化一维数组
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        figure[i*size+j] = static_cast<unsigned char>(distribution(gen));
        // figure[i*size+j] = 1;
        result[i*size+j] = 0;
      }
    }
  }

  ~FigureProcessor() {
    delete[] figure;
    delete[] result;
  }

  // Gaussian filter
  void gaussianFilter() {
    for (size_t i = 1; i < size - 1; ++i) {
      for (size_t j = 1; j < size - 1; ++j) {
        result[i * size + j] =
            (figure[(i - 1) * size + (j - 1)] + 2 * figure[(i - 1) * size + j] +
             figure[(i - 1) * size + (j + 1)] + 2 * figure[i * size + (j - 1)] +
             4 * figure[i * size + j] + 2 * figure[i * size + (j + 1)] +
             figure[(i + 1) * size + (j - 1)] + 2 * figure[(i + 1) * size + j] +
             figure[(i + 1) * size + (j + 1)]) /
            16;
      }
    }

    for (size_t i = 1; i < size - 1; ++i) {
      result[i * size + 0] =
          (figure[(i - 1) * size + 0] + 2 * figure[(i - 1) * size + 0] +
           figure[(i - 1) * size + 1] + 2 * figure[i * size + 0] +
           4 * figure[i * size + 0] + 2 * figure[i * size + 1] +
           figure[(i + 1) * size + 0] + 2 * figure[(i + 1) * size + 0] +
           figure[(i + 1) * size + 1]) /
          16;

      result[i * size + (size - 1)] =
          (figure[(i - 1) * size + (size - 2)] +
           2 * figure[(i - 1) * size + (size - 1)] +
           figure[(i - 1) * size + (size - 1)] +
           2 * figure[i * size + (size - 2)] +
           4 * figure[i * size + (size - 1)] +
           2 * figure[i * size + (size - 1)] +
           figure[(i + 1) * size + (size - 2)] +
           2 * figure[(i + 1) * size + (size - 1)] +
           figure[(i + 1) * size + (size - 1)]) /
          16;
    }

    for (size_t j = 1; j < size - 1; ++j) {
      result[0 * size + j] =
          (figure[0 * size + (j - 1)] + 2 * figure[0 * size + j] +
           figure[0 * size + (j + 1)] + 2 * figure[0 * size + (j - 1)] +
           4 * figure[0 * size + j] + 2 * figure[0 * size + (j + 1)] +
           figure[1 * size + (j - 1)] + 2 * figure[1 * size + j] +
           figure[1 * size + (j + 1)]) /
          16;

      result[(size - 1) * size + j] =
          (figure[(size - 2) * size + (j - 1)] +
           2 * figure[(size - 2) * size + j] +
           figure[(size - 2) * size + (j + 1)] +
           2 * figure[(size - 1) * size + (j - 1)] +
           4 * figure[(size - 1) * size + j] +
           2 * figure[(size - 1) * size + (j + 1)] +
           figure[(size - 1) * size + (j - 1)] +
           2 * figure[(size - 1) * size + j] +
           figure[(size - 1) * size + (j + 1)]) /
          16;
    }

    // 处理四个角点
    result[0 * size + 0] =
        (4 * figure[0 * size + 0] + 2 * figure[0 * size + 1] +
         2 * figure[1 * size + 0] + figure[1 * size + 1]) /
        9;

    result[0 * size + (size - 1)] =
        (4 * figure[0 * size + (size - 1)] +
         2 * figure[0 * size + (size - 2)] +
         2 * figure[1 * size + (size - 1)] +
         figure[1 * size + (size - 2)]) /
        9;

    result[(size - 1) * size + 0] =
        (4 * figure[(size - 1) * size + 0] +
         2 * figure[(size - 1) * size + 1] +
         2 * figure[(size - 2) * size + 0] +
         figure[(size - 2) * size + 1]) /
        9;

    result[(size - 1) * size + (size - 1)] =
        (4 * figure[(size - 1) * size + (size - 1)] +
         2 * figure[(size - 1) * size + (size - 2)] +
         2 * figure[(size - 2) * size + (size - 1)] +
         figure[(size - 2) * size + (size - 2)]) /
        9;
  }

  void powerLawTransformation() {
    constexpr float gamma = 0.5f;
    vector<unsigned char> gammaLUT(256);
    for (int i = 0; i < 256; ++i) {
      gammaLUT[i] = static_cast<unsigned char>(
          255.0f * std::pow(i / 255.0f, gamma) + 0.5f);
    }

    for (size_t i = 0; i < size * size; ++i) {
      if (figure[i] == 0) {
        result[i] = 0;
        continue;
      }
      result[i] = gammaLUT[figure[i]];
    }
  }

  unsigned int calcChecksum() {
    unsigned int sum = 0;
    constexpr size_t mod = 1000000007;
    for (size_t i = 0; i < size * size; ++i) {
      sum += result[i];
      sum %= mod;
    }
    return sum;
  }

  void runBenchmark() {
    auto start = std::chrono::high_resolution_clock::now();
    gaussianFilter();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "gF: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    unsigned int sum = calcChecksum();

    start = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "pow: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";
  }
};

int main(int argc, const char **argv) {
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}
