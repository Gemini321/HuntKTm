#ifndef WRAPPER_H
#define WRAPPER_H

#include "llvm/IR/DerivedTypes.h"
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cuda.h>
#include <cuda_runtime.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Mangler.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/Constant.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Debug.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/ADT/StringRef.h>
#include "passes/analyzer/analyzer.h"


using namespace llvm;
class WrapperPass: public ModulePass {
public:
    static char ID;

    WrapperPass();
    ~WrapperPass();

    virtual bool runOnModule(Module &M) override;
    virtual void getAnalysisUsage(AnalysisUsage &AU) const override;
    void initialization(Module &M);
    
    // create CUDA related function wrappers
    void createCudaMallocWrapper(Module &M);
    void createCudaMallocAsyncWrapper(Module &M);
    void createCudaStreamCreateWrapper(Module &M);
    void createCudaMallocHostWrapper(Module &M);
    void createCudaMemcpyWrapper(Module &M);
    void createCudaMemcpyAsyncWrapper(Module &M);
    void createCudaMemsetWrapper(Module &M);
    void createCudaFreeWrapper(Module &M);
    void createCudaFreeAsyncWrapper(Module &M);
    void createCudaLaunchKernelWrapper(Module &M);
    void createCudaEventCreateWrapper(Module &M);
    void createCudaEventRecordWrapper(Module &M);
    void createCudaStreamWaitEventWrapper(Module &M);

    // replace CUDA functions with wrappers
    void replaceCudaWrapper(IRBuilder<> &builder, CallInst *callInst, FunctionCallee &callee);

    // insert CUDA function wrappers
    // void insertCudaInitialize(Module &M);
    // void insertCudaFinalize(Module &M);
private:
    // wrapper functions
    FunctionCallee cudaMallocWrapper;
    FunctionCallee cudaMallocAsyncWrapper;
    FunctionCallee cudaStreamCreateWrapper;
    FunctionCallee cudaMemcpyWrapper;
    FunctionCallee cudaMemcpyAsyncWrapper;
    FunctionCallee cudaMemsetWrapper;
    FunctionCallee cudaFreeWrapper;
    FunctionCallee cudaFreeAsyncWrapper;
    FunctionCallee cudaLaunchKernelWrapper;
    FunctionCallee cudaEventCreateWrapper;
    FunctionCallee cudaEventRecordWrapper;
    FunctionCallee cudaStreamWaitEventWrapper;
};


#endif
