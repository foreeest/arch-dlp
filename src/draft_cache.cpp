#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>

#include <immintrin.h>

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

    for (size_t j = 1; j < size - 1; ++j) {
      for (size_t i = 1; i < size - 1; ++i) {
        result[i][j] =
            (figure[i - 1][j - 1] + 2 * figure[i - 1][j] +
             figure[i - 1][j + 1] + 2 * figure[i][j - 1] + 4 * figure[i][j] +
             2 * figure[i][j + 1] + figure[i + 1][j - 1] +
             2 * figure[i + 1][j] + figure[i + 1][j + 1]) /
            16;
      }
    }

    for (size_t i = 1; i < size - 1; ++i) {
      result[i][0] =
          (figure[i - 1][0] + 2 * figure[i - 1][0] + figure[i - 1][1] +
           2 * figure[i][0] + 4 * figure[i][0] + 2 * figure[i][1] +
           figure[i + 1][0] + 2 * figure[i + 1][0] + figure[i + 1][1]) /
          16;

      result[i][size - 1] =
          (figure[i - 1][size - 2] + 2 * figure[i - 1][size - 1] +
           figure[i - 1][size - 1] + 2 * figure[i][size - 2] +
           4 * figure[i][size - 1] + 2 * figure[i][size - 1] +
           figure[i + 1][size - 2] + 2 * figure[i + 1][size - 1] +
           figure[i + 1][size - 1]) /
          16;
    }

    for (size_t j = 1; j < size - 1; ++j) {
      result[0][j] =
          (figure[0][j - 1] + 2 * figure[0][j] + figure[0][j + 1] +
           2 * figure[0][j - 1] + 4 * figure[0][j] + 2 * figure[0][j + 1] +
           figure[1][j - 1] + 2 * figure[1][j] + figure[1][j + 1]) /
          16;

      result[size - 1][j] =
          (figure[size - 2][j - 1] + 2 * figure[size - 2][j] +
           figure[size - 2][j + 1] + 2 * figure[size - 1][j - 1] +
           4 * figure[size - 1][j] + 2 * figure[size - 1][j + 1] +
           figure[size - 1][j - 1] + 2 * figure[size - 1][j] +
           figure[size - 1][j + 1]) /
          16;
    }

    // 处理四个角点
    // 左上角
    result[0][0] = (4 * figure[0][0] + 2 * figure[0][1] + 2 * figure[1][0] +
                    figure[1][1]) /
                   9; 

    // 右上角
    result[0][size - 1] = (4 * figure[0][size - 1] + 2 * figure[0][size - 2] +
                           2 * figure[1][size - 1] + figure[1][size - 2]) /
                          9;

    // 左下角
    result[size - 1][0] = (4 * figure[size - 1][0] + 2 * figure[size - 1][1] +
                           2 * figure[size - 2][0] + figure[size - 2][1]) /
                          9;

    // 右下角
    result[size - 1][size - 1] =
        (4 * figure[size - 1][size - 1] + 2 * figure[size - 1][size - 2] +
         2 * figure[size - 2][size - 1] + figure[size - 2][size - 2]) /
        9;
  }

  void gaussianFilterOptimized() {
      for (size_t i = 1; i < size - 1; ++i) {
          for (size_t j = 1; j < size - 1; j += 16) { // 假设每次处理16个像素
              __m256i sum = _mm256_setzero_si256(); // 用于存储加权和

              // 使用AVX2加载相邻的像素值
              __m256i row1 = _mm256_loadu_si256((__m256i*)&figure[i-1][j-1]);
              __m256i row2 = _mm256_loadu_si256((__m256i*)&figure[i][j-1]);
              __m256i row3 = _mm256_loadu_si256((__m256i*)&figure[i+1][j-1]);
              
              // 加权并进行加法运算
              sum = _mm256_add_epi8(sum, row1); // 加权系数是硬编码的，可以按需求调整

              _mm256_storeu_si256((__m256i*)&result[i][j], sum); // 存储结果
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
    
    for (size_t j = 0; j < size; ++j) {
      for (size_t i = 0; i < size; ++i) {
        if(figure[i][j] == 0) {
          result[i][j] = 0;
          continue;
        }
        // LUT 4630ms -> 2272ms
        result[i][j] = gammaLUT[figure[i][j]];
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
    gaussianFilter();
    auto middle = std::chrono::high_resolution_clock::now();

    unsigned int sum = calcChecksum();

    auto middle2 = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    auto end = std::chrono::high_resolution_clock::now();

    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";

    auto milliseconds1 =
        std::chrono::duration_cast<std::chrono::milliseconds>(middle - start);
    auto milliseconds2 = 
        std::chrono::duration_cast<std::chrono::milliseconds>(end - middle2);
    std::cout << "Benchmark time1: " << milliseconds1.count() << " ms\n";
    std::cout << "Benchmark time2: " << milliseconds2.count() << " ms\n";
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
