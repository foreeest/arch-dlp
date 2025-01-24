# SYSU Architeture Final Project

This is a simple implementation of SYSU Architeture final project, it use hardware technique to accelerate image processing(eg. SIMD, OpenMP), see the report for more details.

## Installation

you need to check that your host support AVX2 and SSE4.1, then 
```shell
$ sudo apt-get install -y g++ libomp-dev
```

do the test  
```shell
$ ./runOmpCode.sh before_optimize
$ ./runOmpCode.sh after_optimize
```

or debug with vscode  
or run it in the github codespace  