// v2a LUT OpenMP(4)  
// v2a LUT OpenMP(4) AVX

#include <chrono>
#include <cstddef>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <omp.h>
#include <immintrin.h>
#include <cstring>

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


    void gFOpt() {
        // 并行化外层循环
        #pragma omp parallel for
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
    }

    void gFOpt1() {
        constexpr size_t vecSize = 16; // AVX2 每次处理 16 个字节
        #pragma omp parallel for
        for (size_t i = 1; i < size - 1; ++i) {
            for (size_t j = 1; j + vecSize <= size - 1; j += vecSize) { // 每次处理 16 个像素
                // 加载 figure 数据
                __m128i row1_left  = _mm_loadu_si128((__m128i*)&figure[(i - 1) * size + (j - 1)]);
                __m128i row1_mid   = _mm_loadu_si128((__m128i*)&figure[(i - 1) * size + j]);
                __m128i row1_right = _mm_loadu_si128((__m128i*)&figure[(i - 1) * size + (j + 1)]);

                __m128i row2_left  = _mm_loadu_si128((__m128i*)&figure[i * size + (j - 1)]);
                __m128i row2_mid   = _mm_loadu_si128((__m128i*)&figure[i * size + j]);
                __m128i row2_right = _mm_loadu_si128((__m128i*)&figure[i * size + (j + 1)]);

                __m128i row3_left  = _mm_loadu_si128((__m128i*)&figure[(i + 1) * size + (j - 1)]);
                __m128i row3_mid   = _mm_loadu_si128((__m128i*)&figure[(i + 1) * size + j]);
                __m128i row3_right = _mm_loadu_si128((__m128i*)&figure[(i + 1) * size + (j + 1)]);

                // 扩展为 16 位数据
                __m256i row1_left_16  = _mm256_cvtepu8_epi16(row1_left);
                __m256i row1_mid_16   = _mm256_cvtepu8_epi16(row1_mid);
                __m256i row1_right_16 = _mm256_cvtepu8_epi16(row1_right);

                __m256i row2_left_16  = _mm256_cvtepu8_epi16(row2_left);
                __m256i row2_mid_16   = _mm256_cvtepu8_epi16(row2_mid);
                __m256i row2_right_16 = _mm256_cvtepu8_epi16(row2_right);

                __m256i row3_left_16  = _mm256_cvtepu8_epi16(row3_left);
                __m256i row3_mid_16   = _mm256_cvtepu8_epi16(row3_mid);
                __m256i row3_right_16 = _mm256_cvtepu8_epi16(row3_right);

                // 计算加权和
                __m256i weighted_sum = _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_add_epi16(row1_left_16, row1_right_16),
                        _mm256_add_epi16(_mm256_slli_epi16(row1_mid_16, 1), _mm256_slli_epi16(row2_mid_16, 2))
                    ),
                    _mm256_add_epi16(
                        _mm256_add_epi16(row3_left_16, row3_right_16),
                        _mm256_add_epi16(_mm256_slli_epi16(row2_left_16, 1), _mm256_slli_epi16(row2_right_16, 1))
                    )
                );

                // 平均计算
                __m256i result_16 = _mm256_srli_epi16(weighted_sum, 4);

                // 压缩回 8 位
                __m128i result_8  = _mm_packus_epi16(_mm256_extracti128_si256(result_16, 0), _mm256_extracti128_si256(result_16, 1));

                // 存储结果
                _mm_storeu_si128((__m128i*)&result[i * size + j], result_8);
            }
        }
    }

    void powOpt() {
        constexpr float gamma = 0.5f;
        std::vector<unsigned char> gammaLUT(256);

        // 生成查找表无需并行，开销很小
        for (int i = 0; i < 256; ++i) {
            gammaLUT[i] = static_cast<unsigned char>(
                255.0f * std::pow(i / 255.0f, gamma) + 0.5f);
        }

        // 并行化数组处理
        #pragma omp parallel for
        for (size_t i = 0; i < size * size; ++i) {
            if (figure[i] == 0) {
                result[i] = 0;
                continue;
            }
            result[i] = gammaLUT[figure[i]];
        }
    }

    void powOpt1() {
        constexpr float gamma = 0.5f;
        std::vector<unsigned char> gammaLUT(256);
        for (int i = 0; i < 256; ++i) {
            gammaLUT[i] = static_cast<unsigned char>(255.0f * std::pow(i / 255.0f, gamma) + 0.5f);
        }
        constexpr size_t vecSize = 32; // AVX2 每次处理 32 个字节
        size_t totalElements = size * size;
        // 确保 LUT 数据的连续性和对齐性
        alignas(32) unsigned char lutAligned[256];
        std::memcpy(lutAligned, gammaLUT.data(), 256);

        #pragma omp parallel for
        for (size_t i = 0; i < totalElements; i += vecSize) {
            // 加载 32 字节到 SIMD 寄存器
            __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&figure[i]));
            // 生成掩码：检查 data 中哪些元素为 0
            __m256i zeroVec = _mm256_setzero_si256();
            __m256i mask = _mm256_cmpeq_epi8(data, zeroVec);
            // 数据需要从 8 位扩展到 32 位以适配 i32gather
            __m256i indices = _mm256_cvtepu8_epi32(_mm256_castsi256_si128(data));
            __m256i lutData = _mm256_i32gather_epi32(reinterpret_cast<const int*>(lutAligned), indices, 1);

            __m256i resultVec = _mm256_blendv_epi8(lutData, zeroVec, mask);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result[i]), resultVec);
        }
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
    gFOpt();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "gFOpt: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    gFOpt1();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "gFOpt1: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    // double st1 = omp_get_wtime();
    // gFOpt();
    // double ed1 = omp_get_wtime();
    // std::cout << "gFOpt2: " << (ed1 - st1) << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "pow: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    powOpt();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "powOpt: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    powOpt1();
    end = std::chrono::high_resolution_clock::now();
    std::cout << "powOpt1: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms\n";

    // double st2 = omp_get_wtime();
    // powOpt();
    // double ed2 = omp_get_wtime();
    // std::cout << "powOpt2: " << (ed2 - st2) << " seconds" << std::endl;


    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";
  }
};

int main(int argc, const char **argv) { 
  // omp_set_num_threads(4); // 强制使用 4 个线程
  omp_set_num_threads(2);
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}
