# 图像处理优化

## 初步研究

**基本数据**

- 空跑时基本只在一个CPU上跑满，O0在10s左右(6.2 + 4.6)，O2、O3在2s左右
- 每次跑的都不一样，即使设置了随机种子
- 命令都是`g++ -O0 -std=c++20 -m64 -mavx2 -o "$PROGRAM_NAME" "$PROGRAM_NAME.cpp"`，脚本跑的


程序      |  描述                | 用时           
--- | ---- | -----
cmp0      | origin version，拆开来输出 | 5800 5800
cmp1      | LUT                    | 5200 2000
draft cache | LUT + v2a,列优先       | 2000 1600
try1      | LUT + 分块(企图迎合cache)| 2000 1200
cmp2      | LUT + v2a              | 1700 650(一维vector的话乘2、二维数组1600 800)
try2      | LUT + v2a + AVX2       | 400 80
try3 v1   | LUT + v2a + OpenMP(16) | 260 100 (4thread 450 170; 2thread 900 330)
try3 v2   | LUT + v2a + OpenMP(16) x AVX2| 62 16 (4thread 116 20; 32thread 54 16; 2thread 230 40)
draft_seperate | 分离巻积(基于cmp2) | 
draft_tbb | 

**可能方案**

- cache policy & cache hit\miss
vector怎么存的？不连续  
读写怎么改的？没搞明白  

- AVX2 256bit(simd)
能快8倍？
TODO 数据级并行，再精进一下，研究一下数据对齐；GF可能快不了32倍
no no no 快32倍

- LUT
2  倍

- OpenMP
能快4倍？确实是  

- OpenMP + AVX
多核有对应那么多SIMD可以用吗？  

- MPI + pthread
超算队文档 https://github.com/foreeest/Recruit-23/tree/latest/3-Optimization

- 巻积核分离 + SIMD + OpenMP
https://blog.mangoeffect.net/opencv/convolution-optimization-taking-gaussian-filtering-as-an-example

### 课件

## 存储顺序

- `gen 的顺序是从上到下从左到右即可` 什么意思？  
意思是还是得两个for？

- 数组和vector的存储有区别吗？
3x3打印地址
```
v_2:0x55555556def0 a_2:0x7fffffffd4a0
v_2:0x55555556def4 a_2:0x7fffffffd4a4
v_2:0x55555556def8 a_2:0x7fffffffd4a8
v_2:0x55555556df10 a_2:0x7fffffffd4ac
v_2:0x55555556df14 a_2:0x7fffffffd4b0
v_2:0x55555556df18 a_2:0x7fffffffd4b4
v_2:0x55555556ded0 a_2:0x7fffffffd4b8
v_2:0x55555556ded4 a_2:0x7fffffffd4bc
v_2:0x55555556ded8 a_2:0x7fffffffd4c0
```
看到vector多占4个byte, meta data  

- 跟cache有关系吗？  

16384 x 4byte = 16 x 1024 x 1byte = 16kB
16384 x 16kB = 256MB

my cache
```shell
$ lscpu -C
NAME ONE-SIZE ALL-SIZE WAYS TYPE        LEVEL  SETS PHY-LINE COHERENCY-SIZE
L1d       32K     256K    8 Data            1    64        1             64
L1i       32K     256K    8 Instruction     1    64        1             64
L2         1M       8M    8 Unified         2  2048        1             64
L3        16M      16M   16 Unified         3 16384        1             64
```

改block没有什么用，即把2重循环写成4重，不是很懂cache的具体操作方法

## OpenMP
- 启用编译选项，该选项不影响其它执行时间
- opm的始终和chronos就差2ms可以忽略

为何最终try3 openmp+simd 之后gf还是pow的4倍左右

## 面向github action的优化
本地跑一样的代码
```
gF: 1772 ms
gFOpt: 467 ms
gFOpt1: 116 ms
pow: 649 ms
powOpt: 171 ms
powOpt1: 24 ms
```
github action
```
gF: 2911 ms
gFOpt: 2867 ms
gFOpt1: 594 ms
pow: 1128 ms
powOpt: 1054 ms
powOpt1: 110 ms
```
显然openMP完全无法体现优势

本地和github action跑的try3校验和还不一样，不过try3确实有问题，本地也是每次校验和不同
中间加乘运算确实是占时间的，注释掉快1倍

github action使用姿势不对？
试试github codespace？ 四核有个两倍，比github action好  

## 巻积分离
3x3理论加速比不高，但不知道能否利用其局部性  
直接比较 基本没怎么快
原来1700, 分离是16xx，而且很均衡，横竖都是800多  

试试整个写完  
这个快不了多少  

## OpenCV源码
https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/smooth.cpp
- [ ] 进一步缓存优化  
https://www.cnblogs.com/Imageshop/p/6376028.html
- [ ] 存储对齐？

## tbb

md配置有点问题  

## 细节注意

- [ ] char存像素高斯滤波会溢出吗？
- [ ] 目前的都是开发版，没管GF边缘，这个不影响时间的
- [ ] 校验和在干啥？
- [ ] 代码格式改为一致

2333原始校验和684550983
2222原始校验和683986230
1111原始校验和680660706


## devcontainer
https://www.chenshaowen.com/blog/github-container-registry.html  

## Instruction Latency

- cvtepu8_epi32
    - [ ] 测试对512的支持
    - `-AVX512F`

## OpenMP会打乱执行写入顺序造成错误
出现了一个棘手的问题
OpenMP对GF是没有影响的，对POW有影响  
为什么呢？主要问题应该是store16会覆盖，因为openmp调度完是不按顺序的，
i+1会被i覆盖掉  