// 二维数组 LUT 矩阵分块(没用的)
#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include <immintrin.h>
#include <cstddef>
#include <cstring>

using std::vector;

class FigureProcessor {
private:
  unsigned char** figure;
  unsigned char** result;
  const size_t size;

public:
  FigureProcessor(size_t size, size_t seed = 0) : size(size) {
    // 动态分配内存
    figure = new unsigned char*[size];
    result = new unsigned char*[size];
    for (size_t i = 0; i < size; ++i) {
      figure[i] = new unsigned char[size];
      result[i] = new unsigned char[size];
    }
    
    // !!! Please do not modify the following code !!!
    // 如果你需要修改内存的数据结构，请不要修改初始化的顺序和逻辑
    // 助教可能会通过指定某个初始化seed 的方式来验证你的代码
    // 如果你修改了初始化的顺序，可能会导致你的代码无法通过测试
    std::random_device rd;
    std::mt19937_64 gen(seed == 0 ? rd() : seed);
    std::uniform_int_distribution<unsigned char> distribution(0, 255);
    // !!! ----------------------------------------- !!!

    // 两个数组的初始化在这里，可以改动，但请注意 gen 的顺序是从上到下从左到右即可。
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        figure[i][j] = static_cast<unsigned char>(distribution(gen));
        result[i][j] = 0;
      }
    }
  }

  ~FigureProcessor() {
    // 释放动态分配的内存
    for (size_t i = 0; i < size; ++i) {
      delete[] figure[i];
      delete[] result[i];
    }
    delete[] figure;
    delete[] result;
  }

  // Gaussian filter
  // [[1, 2, 1], [2, 4, 2], [1, 2, 1]] / 16
  //FIXME: Feel free to optimize this function
  //Hint: You can use SIMD instructions to optimize this function
  void gaussianFilter() {
    // 处理内部区域
    for (size_t i = 1; i < size - 1; ++i) {
      for (size_t j = 1; j < size - 1; ++j) {
        result[i][j] =
            (figure[i - 1][j - 1] + 2 * figure[i - 1][j] +
             figure[i - 1][j + 1] + 2 * figure[i][j - 1] + 4 * figure[i][j] +
             2 * figure[i][j + 1] + figure[i + 1][j - 1] +
             2 * figure[i + 1][j] + figure[i + 1][j + 1]) /
            16;
      }
    }
  }

  void gFOpt1() {
    // no faster when bs = 64、1024
    constexpr size_t blockSize = 64; // 假设行大小为 64B，选择适合的块大小
    for (size_t ii = 0; ii < size; ii += blockSize) {
      for (size_t jj = 0; jj < size; jj += blockSize) {
        for (size_t i = ii; i < std::min(ii + blockSize, size); ++i) {
          for (size_t j = jj; j < std::min(jj + blockSize, size); ++j) {
            if(i<1 || j < 1 || i>=size-1 || j>=size-1) continue;
            result[i][j] =
              (figure[i - 1][j - 1] + 2 * figure[i - 1][j] +
              figure[i - 1][j + 1] + 2 * figure[i][j - 1] + 4 * figure[i][j] +
              2 * figure[i][j + 1] + figure[i + 1][j - 1] +
              2 * figure[i + 1][j] + figure[i + 1][j + 1]) /
              16;
          }
        }
      }
    }
  }

  // Power law transformation
  // FIXME: Feel free to optimize this function
  // Hint: LUT to optimize this function?
  void powerLawTransformation() {
    constexpr float gamma = 0.5f;
    std::vector<unsigned char> gammaLUT(256);
    for (int i = 0; i < 256; ++i) {
        gammaLUT[i] = static_cast<unsigned char>(255.0f * std::pow(i / 255.0f, gamma) + 0.5f);
    }
    
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        if(figure[i][j] == 0) {
          result[i][j] = 0;
          continue;
        }
        // LUT 4630ms -> 2272ms
        result[i][j] = gammaLUT[figure[i][j]];
      }
    }
  }

  void powOpt2() {
    constexpr float gamma = 0.5f;
    std::vector<unsigned char> gammaLUT(256);
    for (int i = 0; i < 256; ++i) {
        gammaLUT[i] = static_cast<unsigned char>(255.0f * std::pow(i / 255.0f, gamma) + 0.5f);
    }
    // fucking slower: 800 -> 1100
    constexpr size_t blockSize = 64; // 假设行大小为 64B，选择适合的块大小
    for (size_t ii = 0; ii < size; ii += blockSize) {
      for (size_t jj = 0; jj < size; jj += blockSize) {
        for (size_t i = ii; i < std::min(ii + blockSize, size); ++i) {
          for (size_t j = jj; j < std::min(jj + blockSize, size); ++j) {
            if(figure[i][j] == 0) {
              result[i][j] = 0;
              continue;
            }
            result[i][j] = gammaLUT[figure[i][j]];
          }
        }
      }
    }
  }

  // Run benchmark
  unsigned int calcChecksum() {
    unsigned int sum = 0;
    constexpr size_t mod = 1000000007;
    for (size_t i = 0; i < size; ++i) {
      for (size_t j = 0; j < size; ++j) {
        sum += result[i][j];
        sum %= mod;
      }
    }
    return sum;
  }
  void runBenchmark() {
    auto start = std::chrono::high_resolution_clock::now();
    // gaussianFilter();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "GF: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    unsigned int sum = calcChecksum();

    start = std::chrono::high_resolution_clock::now();
    // gFOpt1();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "gFOpt: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";
    
    start = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "pow: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    // powOpt1();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "powOpt: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";
  }
};

// Main function
// !!! Please do not modify the main function !!!
int main(int argc, const char **argv) {
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}
