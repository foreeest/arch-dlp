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
  unsigned char* figure; // 一维数组
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
        result[i*size+j] = 0;
      }
    }
  }

  ~FigureProcessor() {
    delete[] figure;
    delete[] result;
  }

  void gaussianFilter() {
    // 核心部分，进行优化
    constexpr size_t vecSize = 16; // AVX2 每次处理 16 个字节
    #pragma omp parallel for
    for (size_t i = 1; i < size - 1; ++i) {
        for (size_t j = 1; j + vecSize <= size - 1; j += vecSize) {
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

            // 128bit数据刚好扩展为256bit
            __m256i row1_left_16  = _mm256_cvtepu8_epi16(row1_left);
            __m256i row1_mid_16   = _mm256_cvtepu8_epi16(row1_mid);
            __m256i row1_right_16 = _mm256_cvtepu8_epi16(row1_right);

            __m256i row2_left_16  = _mm256_cvtepu8_epi16(row2_left);
            __m256i row2_mid_16   = _mm256_cvtepu8_epi16(row2_mid);
            __m256i row2_right_16 = _mm256_cvtepu8_epi16(row2_right);

            __m256i row3_left_16  = _mm256_cvtepu8_epi16(row3_left);
            __m256i row3_mid_16   = _mm256_cvtepu8_epi16(row3_mid);
            __m256i row3_right_16 = _mm256_cvtepu8_epi16(row3_right);

            // 计算加权和, 这样调整运算顺序乘法(左移)只需要2次
            __m256i weighted_sum = _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_add_epi16(row1_left_16, row1_right_16),
                    _mm256_add_epi16(row3_left_16, row3_right_16)
                ),
                _mm256_slli_epi16(
                    _mm256_add_epi16(
                        _mm256_add_epi16(row2_left_16, row2_right_16),
                        _mm256_add_epi16(row1_mid_16, row3_mid_16)
                    ),
                    1
                )
            );
            // weighted_sum = _mm256_add_epi16(weighted_sum, _mm256_mullo_epi16(row2_mid_16, _mm256_set1_epi16(4)));
            weighted_sum = _mm256_add_epi16(weighted_sum, _mm256_slli_epi16(row2_mid_16, 2));

            // 平均计算
            __m256i result_16 = _mm256_srli_epi16(weighted_sum, 4);

            // 压缩回 8 位
            __m128i result_8  = _mm_packus_epi16(_mm256_extracti128_si256(result_16, 0), _mm256_extracti128_si256(result_16, 1));

            // 存储结果
            _mm_storeu_si128((__m128i*)&result[i * size + j], result_8);
        }
    }
    // 边缘部分, 数量级少10^3倍，忽略不优化，直接保留；
    // 额外增加一个中心区域的、上述simd step=16处理完余14的部分
    for (size_t i = 1; i < size - 1; ++i){
        for (size_t j = 16369; j < size - 1; ++j){
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
    unsigned int gammaLUT[256];
    for (int i = 0; i < 256; ++i) { // 预先计算
      gammaLUT[i] = static_cast<unsigned int>(
          255.0f * std::pow(i / 255.0f, gamma) + 0.5f);
    }
    constexpr size_t vecSize = 8; 
    size_t totalElements = size * size;
    // 确保 LUT 数据的连续性和对齐性
    alignas(32) unsigned int lutAligned[256];
    std::memcpy(lutAligned, gammaLUT, sizeof(gammaLUT));

    #pragma omp parallel for
    for (size_t i = 0; i < totalElements; i += vecSize) {
      // 加载 8 字节到 SIMD 寄存器； 读16byte，但有效只有8byte，gather是瓶颈；load64没有比128快
      __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&figure[i]));

      // 将 8 位数据扩展为 32 位整数，适配 gammaLUT 查找；取data的低64bit
      __m256i indices = _mm256_cvtepu8_epi32(data);

      // 获取 gammaLUT 中的值； 步长为1byte，因为lutAligned的数据是char
      __m256i lutData = _mm256_i32gather_epi32(reinterpret_cast<const int*>(lutAligned), indices, 4);

      // 给他把8个int压缩回8个char，dst高64位置为0
      // __m128i lutData8 = _mm256_cvtepi32_epi8(lutData); // AVX512，which is not supported

      // SSE is supported
      __m128i lutData16  = _mm_packus_epi32(_mm256_extracti128_si256(lutData, 0), _mm256_extracti128_si256(lutData, 1));
      __m128i zeroVec = _mm_setzero_si128();
      __m128i lutData8 = _mm_packus_epi16(lutData16, zeroVec);

      // 将处理结果存回 `result` 数组；64匹配上步长为8,使openMP仍然正确
      _mm_storeu_si64(reinterpret_cast<__m128i*>(&result[i]), lutData8);
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
    auto middle = std::chrono::high_resolution_clock::now();

    unsigned int sum = calcChecksum();

    auto middle2 = std::chrono::high_resolution_clock::now();
    powerLawTransformation();
    auto end = std::chrono::high_resolution_clock::now();

    sum += calcChecksum();
    sum %= 1000000007;
    std::cout << "Checksum: " << sum << "\n";

    auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(middle - start) +
        std::chrono::duration_cast<std::chrono::milliseconds>(end - middle2);
    std::cout << "Benchmark time: " << milliseconds.count() << " ms\n";
  }
};

int main(int argc, const char **argv) { 
  omp_set_num_threads(4); // 强制使用 4 个线程
  constexpr size_t size = 16384;
  FigureProcessor processor(size, argc > 1 ? std::stoul(argv[1]) : 0);
  processor.runBenchmark();
  return 0;
}
