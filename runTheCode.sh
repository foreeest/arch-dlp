#!/bin/bash

# 检查参数是否提供
if [ $# -eq 0 ]; then
    echo "Usage: $0 <program_name>"
    exit 1
fi

# 获取程序名
PROGRAM_NAME=$1

# 创建 build 文件夹（如果不存在）
BUILD_DIR="./build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi

# 编译指令
g++ -O0 -std=c++20 -m64 -mavx2 -o "./build/$PROGRAM_NAME" "./src/$PROGRAM_NAME.cpp"

# 检查编译是否成功
if [ $? -ne 0 ]; then
    echo "Compilation failed."
    exit 1
fi

# 运行程序
./"$BUILD_DIR/$PROGRAM_NAME" 2333
