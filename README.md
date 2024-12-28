# SYSU Architeture Final Project

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