# HuntKTm

HuntKTm: Hybrid Scheduling and Automatic Management for Efficient Kernel Execution on Modern GPUs

## Requirements

* cmake (version 3.27.7)
* llvm (version 14.0.6)
* cuda (version 12.3)

Note: The operating system we used is Debian 10.2.1 and spark is used to manage softwares with different versions.

## Building

```shell
git clone https://github.com/Gemini321/HuntKTm.git
cd HuntKTm
./build.sh
```

Note: Ensure libstatus.so is on your LD\_LIBRARY\_PATH.

## Compilation

Compiling the benchmarks with script:

```shell
./compile.sh
```

If you want to compile your program manunally, use `libSchedulerPass.so` and `libWrapperPass.so` to transform the LLVM bitcode (either with `opt` or `clang`). Then compile and link runtime library to your program. Refer to the `Makefile` located in `./benchmarks/multi_stream/b1` for guidance.

## Running

Running multi-task or single-task experiment as below:

```shell
# run multi-task experiment
cd driver
./run_multi_task.sh

# run single-task experiment
./run_single_task.sh
```

If you want to run a single task without the scripts we provides, there are two programs to be run: `RuntimeScheduler` and `your_program`. Note that `RuntimeScheduler` should be run before any of your programs.
