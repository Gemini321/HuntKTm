#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <stack>
#include <queue>
#include <map>
#include "passes/analyzer/analyzer.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Local.h"
#include <llvm/Analysis/LoopInfo.h>
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

void printProgram(Module &M) {
    std::error_code EC;
    raw_fd_ostream dest("tmp_opt.ll", EC);
    M.print(dest, nullptr);
}

AnalysisResult::~AnalysisResult() {
    for (auto memObj : memObjList) {
        if (memObj) {
            free(memObj);
        }
    }
}

TaskAnalyzerPass::TaskAnalyzerPass(bool useMemorySchedule): ModulePass(ID), useMemorySchedule(useMemorySchedule) {}

TaskAnalyzerPass::~TaskAnalyzerPass() {}

bool TaskAnalyzerPass::runOnModule(Module &M) {
    bool noWrapper = false;
#ifdef NO_WRAPPER
    noWrapper = true;
#endif
    initialization(M);
    std::error_code EC;
    raw_fd_ostream dest("tmp.ll", EC);
    M.print(dest, nullptr);
    bool success = true;
    Function *mainFunc = M.getFunction("main");
    // dbgs() << "TaskAnalyzerPass: runOnModule\n";
    // collectKernelInvokes(M);
    collectCudaCalls(M);

    if (!useMemorySchedule && noWrapper) {
        while(removeDeadCode(M));
        printProgram(M);
        // dbgs() << "TaskAnalyzerPass: no memory schedule mode\n";
        return true;
    }


    if (result.kernelInvokes.size() == 0) {
        // dbgs() << "kernel invoke number is 0, use lazy mode\n";
        result.isLazy = true;
        insertFakeAddrLookup(M);
        if (mainFunc) {
            insertCudaInitialize(M);
            insertCudaFinalize(M);
        }
        insertCudaLaunchKernelPrepare(M);
        return true;
    }

    // Try to statically analyze the memory requirement and instrument the code
    success = analyzeMemoryRequirement(M);
    success = success ? analyzeThreadRequirement(M) : false;
    success = success ? analyzeIntraSMRequirement(M) : false;
    // result.streamGraph.generateStreamGraphDotFile("stream_graph_before_schedule.dot");
    if (useMemorySchedule) {
        success = success ? scheduleMemoryOperations(M) : false;
    }

    // If no wrapper mode is enabled, return without inserting any runtime functions
    if (noWrapper) {
        while(removeDeadCode(M));
        printProgram(M);
        // dbgs() << "TaskAnalyzerPass: no wrapper mode\n";
        return true;
    }

    success = success ? insertMemoryRequirement(M) : false;

    // If multiple files are compiled, use lazy runtime directly
    if (mainFunc == nullptr) {
        result.isLazy = true;
        // dbgs() << "TaskAnalyzerPass: multiple files are compiled (main function not found), use dynamic analysis\n";
        insertFakeAddrLookup(M);
        insertCudaLaunchKernelPrepare(M);
        return true;
    }

    // Decide to use lazy runtime or not based on the result of static analysis
    if (success) {
        result.isLazy = false;
        // dbgs() << "TaskAnalyzerPass: static analysis successed\n";
    }
    else {
        result.isLazy = true;
        // dbgs() << "TaskAnalyzerPass: static analysis failed, use dynamic analysis instead\n";
        success = insertLazySchedule(M);
        insertFakeAddrLookup(M);
        insertCudaLaunchKernelPrepare(M);
        if (!success) {
            // dbgs() << "cudaTaskScheduleLazy insertion failed, which would be invoked before each kernel launch\n";
        }
    }

    insertCudaInitialize(M);
    insertCudaFinalize(M);
    while(removeDeadCode(M));
    // removeDeadCode(M);
    // dbgs() << "TaskAnalyzerPass: runOnModule finished\n";
    printProgram(M);
    // result.streamGraph.printStreamGraph();
    // result.streamGraph.generateStreamGraphDotFile();
    for (auto &F : M) {
        verifyFunction(F);
    }
    return true;
}

void TaskAnalyzerPass::getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addRequired<LoopInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
}

void TaskAnalyzerPass::initialization(Module &M) {
    createCudaPredictPeakMemory(M);
    createCudaTaskSchedule(M);
    createCudaTaskScheduleLazy(M);
    createCudaLaunchKernelPrepare(M);
    createFakeAddrLookup(M);
    createCudaMallocFunc(M);
    createCudaMallocAsyncFunc(M);
    createCudaFreeAsyncFunc(M);
    createCudaEventCreateFunc(M);
    createCudaEventRecordFunc(M);
    createCudaStreamWaitEventFunc(M);
    createCudaMemPoolCreateFunc(M);
    createCudaMemPoolSetAttribute(M);
    createCudaMemPoolMalloc(M);
    createCudaStreamSetMemPool(M);
}

// Create function `int64_t cudaPredictPeakMemory(int64_t **memory_graph, int64_t *num_node_per_stream, int64_t num_stream)`
void TaskAnalyzerPass::createCudaPredictPeakMemory(Module &M) {
    LLVMContext &context = M.getContext();
    Type *size64Ty = Type::getInt64Ty(context);
    Type *ptrTy = size64Ty->getPointerTo();
    Type *ptrPtrTy = ptrTy->getPointerTo();
    SmallVector<Type*, 3> cudaPredictPeakMemoryArgs = {ptrPtrTy, ptrTy, size64Ty};
    FunctionType *cudaPredictPeakMemoryTy = FunctionType::get(size64Ty, cudaPredictPeakMemoryArgs, false);
    cudaPredictPeakMemory = M.getOrInsertFunction("cudaPredictPeakMemory", cudaPredictPeakMemoryTy);
}

// Create function `cudaError_t cudaTaskSchedule(uint32_t num_block, uint32_t num_threads_per_block, 
// uint64_t total_mem, uint32_t num_stream, bool use_memory_pool)`
void TaskAnalyzerPass::createCudaTaskSchedule(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *size8Ty = Type::getInt8Ty(context);
    Type *size32Ty = Type::getInt32Ty(context);
    Type *size64Ty = Type::getInt64Ty(context);
    SmallVector<Type*, 4> cudaMallocArgs = {size32Ty, size32Ty, size32Ty, size64Ty, size32Ty, size8Ty};
    FunctionType *cudaTaskScheduleTy = FunctionType::get(retTy, cudaMallocArgs, false);
    cudaTaskSchedule = M.getOrInsertFunction("cudaTaskSchedule", cudaTaskScheduleTy);
}

// Create function `cudaError_t cudaTaskScheduleLazy()`
void TaskAnalyzerPass::createCudaTaskScheduleLazy(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    SmallVector<Type*, 3> cudaMallocArgs = {};
    FunctionType *cudaTaskScheduleLazyTy = FunctionType::get(retTy, cudaMallocArgs, false);
    cudaTaskScheduleLazy = M.getOrInsertFunction("cudaTaskScheduleLazy", cudaTaskScheduleLazyTy);
}

// Create function `void cudaLaunchKernelPrepare()`
void TaskAnalyzerPass::createCudaLaunchKernelPrepare(Module &M) {
    LLVMContext &context = M.getContext();
    Type *size64Ty = Type::getInt64Ty(context);
    Type *size32Ty = Type::getInt32Ty(context);
    Type *retTy = Type::getVoidTy(context);
    
    SmallVector<Type*> cudaLaunchKernelPrepareArgs = {size64Ty, size32Ty, size64Ty, size32Ty};
    FunctionType *cudaLaunchKernelPrepareTy = FunctionType::get(retTy, cudaLaunchKernelPrepareArgs, false);
    cudaLaunchKernelPrepare = M.getOrInsertFunction("cudaLaunchKernelPrepare", cudaLaunchKernelPrepareTy);
}

// Create function `void *fakeAddrLookup(void *addr)`
void TaskAnalyzerPass::createFakeAddrLookup(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt8PtrTy(context);
    Type *ptrTy = Type::getInt8PtrTy(context);
    SmallVector<Type*, 1> fakeAddrLookupArgs = {ptrTy};
    FunctionType *fakeAddrLookupTy = FunctionType::get(retTy, fakeAddrLookupArgs, false);
    fakeAddrLookup = M.getOrInsertFunction("fakeAddrLookup", fakeAddrLookupTy);
}

// Create function `cudaError_t cudaMalloc(size_t *devPtr, size_t size)`
void TaskAnalyzerPass::createCudaMallocFunc(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context)->getPointerTo();
    Type *sizeTy = Type::getInt64Ty(context);
    SmallVector<Type*, 2> cudaMallocArgs = {voidPtrTy, sizeTy};
    FunctionType *cudaMallocTy = FunctionType::get(retTy, cudaMallocArgs, false);
    cudaMallocFunc = M.getOrInsertFunction("cudaMalloc", cudaMallocTy);
}

// Create function `cudaError_t cudaMallocAsync(size_t *devPtr, size_t size, cudaStream_t stream)`
void TaskAnalyzerPass::createCudaMallocAsyncFunc(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context)->getPointerTo();
    Type *sizeTy = Type::getInt64Ty(context);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);
    SmallVector<Type*, 3> cudaMallocAsyncArgs = {voidPtrTy, sizeTy, streamStructPtTy};
    FunctionType *cudaMallocAsyncTy = FunctionType::get(retTy, cudaMallocAsyncArgs, false);
    cudaMallocAsyncFunc = M.getOrInsertFunction("cudaMallocAsync", cudaMallocAsyncTy);
}

// Create function `cudaError_t cudaFreeAsync(size_t devPtr, cudaStream_t stream)`
void TaskAnalyzerPass::createCudaFreeAsyncFunc(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);
    SmallVector<Type*, 2> cudaFreeAsyncArgs = {voidPtrTy, streamStructPtTy};
    FunctionType *cudaFreeAsyncTy = FunctionType::get(retTy, cudaFreeAsyncArgs, false);
    cudaFreeAsyncFunc = M.getOrInsertFunction("cudaFreeAsync", cudaFreeAsyncTy);
}

// Create function `cudaError_t cudaEventCreate(cudaEvent_t *event)`
void TaskAnalyzerPass::createCudaEventCreateFunc(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    StructType *eventStructTy = StructType::getTypeByName(context, "struct.CUevent_st");
    // struct.CUevent_st is not defined in the program
    if (eventStructTy == nullptr) {
        eventStructTy = StructType::create(context, "struct.CUevent_st");
    }
    Type *eventStructPtTy = PointerType::getUnqual(eventStructTy);
    eventStructPtTy = PointerType::getUnqual(eventStructPtTy);

    SmallVector<Type*, 1> cudaEventCreateArgs = {eventStructPtTy};
    FunctionType *cudaEventCreateTy = FunctionType::get(retTy, cudaEventCreateArgs, false);
    cudaEventCreateFunc = M.getOrInsertFunction("cudaEventCreate", cudaEventCreateTy);
    // dbgs() << "cudaEventCreateFuncTy: " << *(cudaEventCreateFunc.getFunctionType()) << "\n";
}

// Create function `cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)`
void TaskAnalyzerPass::createCudaEventRecordFunc(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    StructType *eventStructTy = StructType::getTypeByName(context, "struct.CUevent_st");
    // struct.CUevent_st is not defined in the program
    if (eventStructTy == nullptr) {
        eventStructTy = StructType::create(context, "struct.CUevent_st");
    }
    Type *eventStructPtTy = PointerType::getUnqual(eventStructTy);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);

    SmallVector<Type*, 2> cudaEventRecordArgs = {eventStructPtTy, streamStructPtTy};
    FunctionType *cudaEventRecordTy = FunctionType::get(retTy, cudaEventRecordArgs, false);
    cudaEventRecordFunc = M.getOrInsertFunction("cudaEventRecord", cudaEventRecordTy);
}

// Create function `cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)`
void TaskAnalyzerPass::createCudaStreamWaitEventFunc(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    StructType *eventStructTy = StructType::getTypeByName(context, "struct.CUevent_st");
    // struct.CUevent_st is not defined in the program
    if (eventStructTy == nullptr) {
        eventStructTy = StructType::create(context, "struct.CUevent_st");
    }
    Type *eventStructPtTy = PointerType::getUnqual(eventStructTy);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);
    Type *intTy = Type::getInt32Ty(context);

    SmallVector<Type*, 3> cudaStreamWaitEventArgs = {streamStructPtTy, eventStructPtTy, intTy};
    FunctionType *cudaStreamWaitEventTy = FunctionType::get(retTy, cudaStreamWaitEventArgs, false);
    cudaStreamWaitEventFunc = M.getOrInsertFunction("cudaStreamWaitEvent", cudaStreamWaitEventTy);
}

// Create function `cudaError_t cudaMemPoolCreate(cudaMemPool_t *memPool, const struct cudaMemPoolProps *props)`
void TaskAnalyzerPass::createCudaMemPoolCreateFunc(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    StructType *memPoolStructTy = StructType::getTypeByName(context, "struct.cuMemPool_st");
    if (memPoolStructTy == nullptr) {
        memPoolStructTy = StructType::create(context, "struct.cuMemPool_st");
    }
    Type *memPoolStructPtTy = PointerType::getUnqual(memPoolStructTy);
    StructType *propsStructTy = StructType::getTypeByName(context, "struct.cuMemPoolProps_st");
    if (propsStructTy == nullptr) {
        propsStructTy = StructType::create(context, "struct.cuMemPoolProps_st");
    }
    Type *propsStructPtTy = PointerType::getUnqual(propsStructTy);

    SmallVector<Type*, 2> cudaMemPoolCreateArgs = {memPoolStructPtTy, propsStructPtTy};
    FunctionType *cudaMemPoolCreateTy = FunctionType::get(retTy, cudaMemPoolCreateArgs, false);
    cudaMemPoolCreateFunc = M.getOrInsertFunction("cudaMemPoolCreate", cudaMemPoolCreateTy);
}

// Create function `cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void *value)`
void TaskAnalyzerPass::createCudaMemPoolSetAttribute(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    StructType *memPoolStructTy = StructType::getTypeByName(context, "struct.cuMemPool_st");
    if (memPoolStructTy == nullptr) {
        memPoolStructTy = StructType::create(context, "struct.cuMemPool_st");
    }
    Type *memPoolStructPtTy = PointerType::getUnqual(memPoolStructTy);
    Type *intTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);

    SmallVector<Type*, 3> cudaMemPoolSetAttributeArgs = {memPoolStructPtTy, intTy, voidPtrTy};
    FunctionType *cudaMemPoolSetAttributeTy = FunctionType::get(retTy, cudaMemPoolSetAttributeArgs, false);
    cudaMemPoolSetAttributeFunc = M.getOrInsertFunction("cudaMemPoolSetAttribute", cudaMemPoolSetAttributeTy);
}

// Create function `cudaError_t cudaMemPoolMalloc(cudaMemPool_t memPool, void **ptr, size_t size)`
void TaskAnalyzerPass::createCudaMemPoolMalloc(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    StructType *memPoolStructTy = StructType::getTypeByName(context, "struct.cuMemPool_st");
    if (memPoolStructTy == nullptr) {
        memPoolStructTy = StructType::create(context, "struct.cuMemPool_st");
    }
    Type *memPoolStructPtTy = PointerType::getUnqual(memPoolStructTy);
    Type *voidPtrPtrTy = PointerType::getUnqual(Type::getInt8PtrTy(context));
    Type *sizeTy = Type::getInt64Ty(context);

    SmallVector<Type*, 3> cudaMemPoolMallocArgs = {memPoolStructPtTy, voidPtrPtrTy, sizeTy};
    FunctionType *cudaMemPoolMallocTy = FunctionType::get(retTy, cudaMemPoolMallocArgs, false);
    cudaMemPoolMallocFunc = M.getOrInsertFunction("cudaMemPoolMalloc", cudaMemPoolMallocTy);
}

// Create function `cudaError_t cudaStreamSetMemPool(cudaStream_t stream, cudaMemPool_t memPool)`
void TaskAnalyzerPass::createCudaStreamSetMemPool(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);
    StructType *memPoolStructTy = StructType::getTypeByName(context, "struct.cuMemPool_st");
    if (memPoolStructTy == nullptr) {
        memPoolStructTy = StructType::create(context, "struct.cuMemPool_st");
    }
    Type *memPoolStructPtTy = PointerType::getUnqual(memPoolStructTy);

    SmallVector<Type*, 2> cudaStreamSetMemPoolArgs = {streamStructPtTy, memPoolStructPtTy};
    FunctionType *cudaStreamSetMemPoolTy = FunctionType::get(retTy, cudaStreamSetMemPoolArgs, false);
    cudaStreamSetMemPoolFunc = M.getOrInsertFunction("cudaStreamSetMemPool", cudaStreamSetMemPoolTy);
}

/***********************************************************************************
 *                               Analysis Functions
 ***********************************************************************************
 */

// Collect kernel invokes and build stream graph
void TaskAnalyzerPass::collectKernelInvokes(Module &M) {
    assert(false && "Deprecated");
}

// Collect kernel invokes, CUDA calls and build stream graph
void TaskAnalyzerPass::collectCudaCalls(Module &M) {
    // dbgs() << "TaskAnalyzerPass: collectCudaCalls\n";
    result.streamGraph.setResult(&result);
    // Record cudaStreams
    for (auto &globalValue : M.getGlobalList()) {
        if (auto *GV = dyn_cast<GlobalVariable>(&globalValue)) {
            if (GV->getName().contains("mzw_s")) {
                result.streamGraph.addStream(GV);
            }
        }
    }
    // dbgs() << "Stream number: " << result.streamGraph.getNumStream() << "\n";

    // Collect kernel invokes and CUDA calls
    for (auto &F : M) {
        if (F.isDeclaration()) {
            continue;
        }
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *CI = dyn_cast<CallInst>(&I)) {
                    // Skip indirect calls
                    if (CI->getCalledFunction() == nullptr) {
                        continue;
                    }
                    // Kernel invoke
                    if (CI->getCalledFunction()->getName().contains("__cudaPushCallConfiguration")) {
                        Instruction *nextInst = CI->getNextNonDebugInstruction();
                        int idx = 0;
                        // Skip llvm intrinsic calls and find the next kernel invoke
                        while (!isa<CallInst>(nextInst) 
                            || dyn_cast<CallInst>(nextInst)->getCalledFunction()->getName().contains("llvm") 
                            || dyn_cast<CallInst>(nextInst)->getCalledFunction()->getName().contains("cudaEventRecord")
                            || dyn_cast<CallInst>(nextInst)->getCalledFunction()->getName().contains("cudaEventCreate")) {
                            if (isa<ICmpInst>(nextInst)) {
                                idx = 1 - dyn_cast<ICmpInst>(nextInst)->isTrueWhenEqual();
                                nextInst = nextInst->getNextNonDebugInstruction();
                            }
                            else if (isa<BranchInst>(nextInst)) {
                                nextInst = dyn_cast<BranchInst>(nextInst)->getSuccessor(idx)->getFirstNonPHIOrDbg();
                            }
                            else {
                                nextInst = nextInst->getNextNonDebugInstruction();
                            }
                        }
                        assert(nextInst != nullptr && "No kernel invoke found");

                        // Collect kernel invoke instruction and its thread size (gridDim and blockDim)
                        CallInst *invokeInstr = dyn_cast<CallInst>(nextInst);
                        GlobalVariable *stream = result.streamGraph.retriveStream(CI->getArgOperand(5));
                        // assert(stream != nullptr && "cuStream is not found when kernel invoke collecting");
                        if (stream != nullptr) {
                            StreamGraphNode *node = new StreamGraphNode(invokeInstr, KERNEL_INVOKE, stream);
                            result.streamGraph.addNode(node);
                        }
                        result.kernelInvokeMap[invokeInstr] = new KernelInvoke(invokeInstr, CI, stream);

                        result.kernelInvokes.push_back(invokeInstr);
                        result.threadSizeMap[invokeInstr] = {
                            CI->getArgOperand(0), CI->getArgOperand(1), CI->getArgOperand(2), CI->getArgOperand(3)};
                    }
                    // CUDA calls
                    else if (CI->getCalledFunction()->getName().contains("cudaSetDevice")) {
                        // assert(false && "cudaSetDevice is not supported");
                        // dbgs() << "Warning: cudaSetDevice is used\n";
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaMalloc")) {
                        result.cudaCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaMemcpyAsync")) {
                        GlobalVariable *stream = result.streamGraph.retriveStream(CI->getArgOperand(4));
                        StreamGraphMemNode *node = new StreamGraphMemNode(CI, stream);
                        result.streamGraph.addNode(node);
                        result.cudaCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaMemcpy")) {
                        result.cudaCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaMemset")) {
                        result.cudaCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaFree")) {
                        result.cudaCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaStreamCreate")) {
                        result.cudaStreamCreateCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaEventCreate")) {
                        AllocaInst *event = result.streamGraph.retriveEvent(CI->getArgOperand(0));
                        StreamGraphEdge *edge = new StreamGraphEdge(event);
                        result.streamGraph.addEdge(edge);
                        result.cudaEventCreateCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaEventRecord")) {
                        AllocaInst *event = result.streamGraph.retriveEvent(CI->getArgOperand(0));
                        GlobalVariable *stream = result.streamGraph.retriveStream(CI->getArgOperand(1));
                        assert(event != nullptr && stream != nullptr && "event or stream in cudaEventRecord is not found");
                        StreamGraphEdge *edge = result.streamGraph.getEdge(event);
                        // setSrc automatically append edge to srcNode
                        edge->setSrc(result.streamGraph.getStreamBackNode(stream));
                    }
                    // @TODO: delete wait_event node when successor node comes
                    else if (CI->getCalledFunction()->getName().contains("cudaStreamWaitEvent")) {
                        GlobalVariable *stream = result.streamGraph.retriveStream(CI->getArgOperand(0));
                        AllocaInst *event = result.streamGraph.retriveEvent(CI->getArgOperand(1));
                        assert(event != nullptr && stream != nullptr && "event or stream in cudaStreamWaitEvent is not found");
                        StreamGraphEdge *edge = result.streamGraph.getEdge(event);
                        StreamGraphNode *node = new StreamGraphNode(CI, CUDA_STREAM_WAIT_EVENT, stream);
                        result.streamGraph.addNode(node);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaDeviceSynchronize")) {
                        result.cudaCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cudaLaunchKernel")) {
                        result.cudaCalls.push_back(CI);
                    }
                    else if (CI->getCalledFunction()->getName().contains("cuda")) {
                        // dbgs() << "Untracked CUDA call: " << CI->getCalledFunction()->getName() << "\n";
                    }
                }
            }
        }
    }

    for (auto &invokeObj : result.kernelInvokeMap) {
        // dbgs() << "Kernel invoke: " << invokeObj.first->getCalledFunction()->getName() << "\n";
    }
    for (auto *CI : result.cudaCalls) {
        // dbgs() << "CUDA call: " << CI->getCalledFunction()->getName() << "\n";
    }
    result.streamGraph.printStreamGraph();
    result.streamGraph.printStreamEdge();
}

// Return whether the argument is related to memory allocation
bool TaskAnalyzerPass::isArgMallocRelated(Value *arg) {
    // Skip non-pointer arguments
    if (arg->getType()->isPointerTy() == false) {
        return false;
    }
    // Skip non-pointer arguments allocation
    else if (auto *AI = dyn_cast<AllocaInst>(arg)) {
        if (AI->getAllocatedType()->isPointerTy() == false) {
            return false;
        }
    }
    return true;
}


// Search for the malloc call that allocates the memory for kernel invoke argument
CallInst *TaskAnalyzerPass::searchMallocByArg(Value *arg) {
    // dbgs() << "TaskAnalyzerPass: analyzeInvokeArg for " << *arg << "\n";
    assert(arg->getType()->isPointerTy() && "Malloc related argument must be a pointer");
    // // dbgs() << "Pointer type: " << *arg << "\n";
    if (auto *LI = dyn_cast<LoadInst>(arg)) {
        // // dbgs() << "LoadInst getPointerOperand(): " << *(LI->getPointerOperand()) << "\n";
        if (auto *AI = dyn_cast<AllocaInst>(LI->getPointerOperand())) {
            for (auto user: AI->users()) {
                // // dbgs() << "User: " << *user << "\n";
                // Directly cudaMalloc the pointer
                if (auto *CI = dyn_cast<CallInst>(user)) {
                    auto name = getDemangledName(CI->getCalledFunction()->getName().str());
                    if (name == "cudaMalloc") {
                        return CI;
                    }
                }
                // cudaMalloc the pointer after bitcast
                else if (auto *BI = dyn_cast<BitCastInst>(user)) {
                    // // dbgs() << "Find BitCastInst\n";
                    for (auto user: BI->users()) {
                        // // dbgs() << "User: " << *user << "\n";
                        if (auto *CI = dyn_cast<CallInst>(user)) {
                            auto name = getDemangledName(CI->getCalledFunction()->getName().str());
                            if (name == "cudaMalloc") {
                                return CI;
                            }
                        }
                    }
                }
            }
        }
    }
    // If cudaMalloc is not found, return nullptr and set lazy mode
    return nullptr;
}

// Analyze the memory requirement of each kernel invoke
bool TaskAnalyzerPass::analyzeMemoryRequirement(Module &M) {
    // dbgs() << "TaskAnalyzerPass: analyzeMemoryRequirement\n";
    StringRef funcName = result.kernelInvokes.empty() ? 
        "" : result.kernelInvokes[0]->getFunction()->getName();
    for (auto *CI: result.kernelInvokes) {
        if (CI->getFunction()->getName() != funcName) {
            // dbgs() << "Kernel invokes are not in the same function, memory analysis failed\n";
            return false;
        }
    }

    // Search cudaMalloc for each kernel invoke
    for (auto *CI : result.kernelInvokes) {
        // dbgs() << "Analyzing kernel invoke: " << *CI << "\n";
        for (auto &arg : CI->args()) {
            // // dbgs() << "Arg: " << arg << "\n";
            if (!isArgMallocRelated(arg.get())) {
                continue;
            }
            if (auto *malloc = searchMallocByArg(arg.get())) {
                // dbgs() << "Malloc found: " << *malloc << " for " << *CI << "\n";
                result.mallocMap[CI].push_back(malloc);
                MemoryObj *memObj = new MemoryObj(malloc);
                
                if (result.mallocSizeMap.find(memObj->getTarget()) == result.mallocSizeMap.end()) {
                    result.mallocSizeMap[memObj->getTarget()] = memObj->getSize();
                    result.memObjList.push_back(memObj);
                    // dbgs() << "MemObj is created: " << *malloc << ", target: " << *memObj->getTarget() << ", size: " << *memObj->getSize() << "\n";
                }
                else {
                    for (auto memObjCreated : result.memObjList) {
                        if (memObjCreated->getTarget() == memObj->getTarget()) {
                            free(memObj);
                            memObj = memObjCreated;
                            // dbgs() << "Malloc size of " << *memObj->getTarget() << " already exists, size: " << *memObj->getSize() << "\n";
                            break;
                        }
                    }
                }
                result.kernelInvokeMap[CI]->appendMemoryObj(memObj);
                memObj->appendKernelInvoke(result.kernelInvokeMap[CI]);
            }
            else {
                // dbgs() << "Malloc not found for " << *arg << ", set lazy mode\n";
                return false;
            }
        }
    }

    // Analyze memory requirements of stream graph
    for (auto node : result.streamGraph.getSequenceGraph()) {
        if (node->kind == CUDA_MEMNODE) {
            StreamGraphMemNode *memNode = dyn_cast<StreamGraphMemNode>(node);
            memNode->retriveMemcpyKind();
            memNode->retriveMemoryObj(result.memObjList);
            // dbgs() << "memNode: " << *(memNode->call) << "\n";
            // dbgs() << "    kind: " << memNode->memcpyKind << "\n";
            // dbgs() << "    edges:\n";
            for (auto edge : memNode->outEdges) {
                if (edge->event) {
                    // dbgs() << "    " << *(edge->event) << "\n";
                }
            }
            assert(memNode->memoryObj && "Do not found memoryObj");
            // dbgs() << "    memoryObj malloc: " << *(memNode->memoryObj->getMalloc()) << "\n";
            for (auto instr : memNode->memoryObj->getMallocRelatedInstrList()) {
                // dbgs() << "    memoryObj malloc related: " << *instr << "\n";
            }
            // dbgs() << "    memoryObj free: " << *(memNode->memoryObj->getFree()) << "\n";
            for (auto instr : memNode->memoryObj->getFreeRelatedInstrList()) {
                // dbgs() << "    memoryObj free related: " << *instr << "\n";
            }
            for (auto memcpy : memNode->memoryObj->getMemcpyList()) {
                // dbgs() << "    memoryObj memcpy: " << *memcpy << "\n";
                for (auto instr : memNode->memoryObj->getMemcpyRelatedInstrList(memcpy)) {
                    // dbgs() << "    memoryObj free related: " << *instr << "\n";
                }
            }
            for (auto kernelInvoke : memNode->memoryObj->getKernelInvokeList()) {
                // dbgs() << "    memoryObj related kernelInvoke: " << *kernelInvoke->getInvoke() << "\n";
            }
        }
    }
    return true;
}

// Analyze the thread requirement of each kernel invoke
bool TaskAnalyzerPass::analyzeThreadRequirement(Module &M) {
    for (auto kernelInvokePair : result.kernelInvokeMap) {
        KernelInvoke *invokeInfo = kernelInvokePair.second;
        CallInst *cudaPush = invokeInfo->cudaPush;

        // Find grid dim ctor
        Instruction *instr = dyn_cast<Instruction>(cudaPush->getArgOperand(0));
        assert(instr && "GridDim ctor not found");
        LoadInst *load = dyn_cast<LoadInst>(instr);
        assert(load && "xxx");
        BitCastInst *bitcast = dyn_cast<BitCastInst>(load->getOperand(0));
        assert(bitcast && isDim3Struct(bitcast->getSrcTy()) && "yyy");
        Value *dim3 = bitcast->getOperand(0);
        invokeInfo->gridDim.setDim3(dim3);
        invokeInfo->gridDim.retriveCtor();
        invokeInfo->gridDim.retriveDim();

        // Find block dim ctor
        instr = dyn_cast<Instruction>(cudaPush->getArgOperand(2));
        assert(instr && "GridDim ctor not found");
        load = dyn_cast<LoadInst>(instr);
        assert(load && "xxx");
        bitcast = dyn_cast<BitCastInst>(load->getOperand(0));
        assert(bitcast && isDim3Struct(bitcast->getSrcTy()) && "yyy");
        dim3 = bitcast->getOperand(0);
        invokeInfo->blockDim.setDim3(dim3);
        invokeInfo->blockDim.retriveCtor();
        invokeInfo->blockDim.retriveDim();

        SmallVector<Value *, 3> gridArgs = invokeInfo->gridDim.getArgs();
        SmallVector<Value *, 3> blockArgs = invokeInfo->blockDim.getArgs();
        // dbgs() << "thread requirement of " << *invokeInfo->getInvoke() << ":\n";
        // dbgs() << "GridDim Ctor " << *invokeInfo->gridDim.ctor << ": " << *gridArgs[0] << ", " << *gridArgs[1] << ", " << *gridArgs[2] << "\n";
        // dbgs() << "BlockDim Ctor " << *invokeInfo->blockDim.ctor << ": " << *blockArgs[0] << ", " << *blockArgs[1] << ", " << *blockArgs[2] << "\n";
    }
    return true;
}

// Analyze intra-SM resource requirement of each kernel invoke
// e.g. register usage, shared memory usage...
bool TaskAnalyzerPass::analyzeIntraSMRequirement(Module &M) {
    // dbgs() << "TaskAnalyzerPass: analyzeIntraSMRequirement\n";
    std::ifstream kernelInfoFile(kernelInfoFileName);
    if (!kernelInfoFile.is_open()) {
        assert(false && ("Failed to open kernel info file: " + kernelInfoFileName).c_str());
    }

    std::regex functionRegex(R"(_Z[\w\d]+)");
    std::regex registerRegex(R"(Used (\d+) registers)");
    std::regex sharedMemoryRegex(R"((\d+) bytes smem)");
    std::map<std::string, std::pair<int, int>> kernelInfo;
    std::string line;
    std::string currentFunction;
    while (std::getline(kernelInfoFile, line)) {
        std::smatch match;
        // Extract function name
        if (std::regex_search(line, match, functionRegex)) {
            currentFunction = match.str();
        }
        if (currentFunction == "" || kernelInfo.count(currentFunction)) {
            continue;
        }
        // Registers usage
        if (std::regex_search(line, match, registerRegex)) {
            int registers = std::stoi(match[1]);
            kernelInfo[currentFunction].first = registers;
        }
        // Shared memory usage
        if (std::regex_search(line, match, sharedMemoryRegex)) {
            int sharedMemory = std::stoi(match[1]);
            kernelInfo[currentFunction].second = sharedMemory;
        }
    }
    for (auto kernelInvokePair : result.kernelInvokeMap) {
        KernelInvoke *invokeInfo = kernelInvokePair.second;
        StringRef funcName = invokeInfo->invoke->getCalledFunction()->getName();

        if (kernelInfo.count(funcName.str())) {
            invokeInfo->numRegister = kernelInfo[funcName.str()].first;
            invokeInfo->numSharedMem = kernelInfo[funcName.str()].second;
        }
        else { // Set default value for not found kernels
            // dbgs() << "[Warning]: Kernel info not found for " << funcName << "\n";
            invokeInfo->numRegister = 64;
            invokeInfo->numSharedMem = 0;
        }
    }

    kernelInfoFile.close();
    return true;
}

// Find the earliest point to move cudaMalloc & cudaMemcpyAsync (H2D)
// We try to find a node as the insert point that has minimal synchronization with other nodes in invokeNodeList.
bool TaskAnalyzerPass::postponeMemoryOperations(MemoryObj *memObj, 
    SmallVector<StreamGraphNode *> invokeNodeList, Module &M) {
    StreamGraphNode *earliestNode = nullptr;
    KernelInvoke *earliestInvoke = nullptr;
    Instruction *insertPoint = nullptr;
    SmallVector<StreamGraphNode *> H2DSyncNodeList = invokeNodeList;
    CallInst *H2DMemcpyCall = nullptr;
    StreamGraphMemNode *H2DMemNode = nullptr;
    for (auto memcpy : memObj->getMemcpyList()) {
        cudaMemcpyKind kind = (cudaMemcpyKind)dyn_cast<ConstantInt>(memcpy->getArgOperand(3))->getZExtValue();
        if (kind == cudaMemcpyHostToDevice) {
            H2DMemcpyCall = memcpy;
            H2DMemNode = dyn_cast<StreamGraphMemNode>(result.streamGraph.getNode(H2DMemcpyCall));
            break;
        }
    }
    // Sort invokeNodeList by the order in sequential graph
    std::sort(invokeNodeList.begin(), invokeNodeList.end(), 
        [&](StreamGraphNode *a, StreamGraphNode *b) {
            return result.streamGraph.getSequenceGraphIndex(a) < result.streamGraph.getSequenceGraphIndex(b);
        });

    if (invokeNodeList.size() == 0) {
        // dbgs() << "No invoke node found, postpone failed\n";
        assert(false && "No invoke node found, postpone failed");
    }
    earliestNode = invokeNodeList[0];
    for (int j = 1; j < invokeNodeList.size(); j ++) {
        if (!result.streamGraph.isDependent(earliestNode, invokeNodeList[j])) {
            // if node j->i is satisfied, should not choose node i when postponing
            if (result.streamGraph.isDependent(invokeNodeList[j], earliestNode)) {
                H2DSyncNodeList = invokeNodeList;
                assert(false && "Order in sequential graph and stream graph can not be satisfied simultaneously");
            }
            H2DSyncNodeList.push_back(invokeNodeList[j]);
        }
    }
    earliestInvoke = result.getKernelInvoke(earliestNode->call);
    insertPoint = earliestInvoke->getCudaPush();
    // dbgs() << "Earliest node: " << *(earliestNode->call) << "\n";
    if (H2DMemcpyCall) {
        // dbgs() << "cudaMemcpyAsync (H2D): " << *H2DMemcpyCall << "\n";
    }
    // dbgs() << "New sync list:\n";
    for (auto node : H2DSyncNodeList) {
        // dbgs() << *node->call << "\n";
    }
    // Move StreamGraphMemNode before the earliest insert point
    // dbgs() << "Insert point when postponing: " << *insertPoint << "\n";
    if (H2DMemNode) {
        H2DMemNode->setMemNodeKind(CUDA_MALLOC_MEMCPY);
        // dbgs() << "Edges of H2DMemNode before moving:\n";
        for (auto edge : H2DMemNode->outEdges) {
            edge->printDst();
        }
        // dbgs() << "Move from stream: " << H2DMemNode->stream->getName() << " to stream: " << earliestNode->stream->getName() << "\n";
        result.streamGraph.moveNodeBefore(H2DMemNode, earliestNode);
        // dbgs() << "Edges of H2DMemNode:\n";
        for (auto edge : H2DMemNode->outEdges) {
            edge->printDst();
        }
        // dbgs() << "Move existing node " << *(H2DMemNode->call) << " before " << *(earliestNode->call) << "\n";
    }
    else {
        H2DMemNode = new StreamGraphMemNode(nullptr, earliestNode->stream);
        H2DMemNode->setMemoryObj(memObj);
        H2DMemNode->setMemcpyKind(cudaMemcpyHostToDevice);
        H2DMemNode->setMemNodeKind(CUDA_MALLOC);
        result.streamGraph.addNode(H2DMemNode);
        result.streamGraph.moveNodeBefore(H2DMemNode, earliestNode);
        // dbgs() << "Move new node " << *(H2DMemNode->memoryObj->malloc) << " before " << *(earliestNode->call) << "\n";
    }
    
    // Move instructions of node before the earliest insert point
    auto loadStream = loadStreamBefore(H2DMemNode->stream, insertPoint, M.getContext());
    std::map<Instruction *, Instruction *> cloneMap;
    // dbgs() << "Moving cudaMalloc related instructions (" << memObj->getMallocRelatedInstrList().size() << " instr) before " << *(earliestNode->call) << "\n";
    for (auto I: memObj->getMallocRelatedInstrList()) {
        // dbgs() << *I << "\n";
    }
    
    SmallVector<Instruction *> mallocRelatedInstrList;
    for (auto originalInstr : memObj->getMallocRelatedInstrList()) {
        Instruction *instr = nullptr;
        if (originalInstr == memObj->getMalloc()) {
            instr = originalInstr;
            instr->moveBefore(insertPoint);
        }
        else {
            instr = originalInstr->clone();
            cloneMap[originalInstr] = instr;
            instr->insertBefore(insertPoint);
            // dbgs() << "Clone instruction: " << *originalInstr << " to " << *instr << "\n";
        }
        mallocRelatedInstrList.push_back(instr);
        
        // Replace uses of original instruction with cloned instruction
        for (auto &use : instr->operands()) {
            if (auto *useInst = dyn_cast<Instruction>(use)) {
                if (cloneMap.find(useInst) != cloneMap.end()) {
                    instr->replaceUsesOfWith(useInst, cloneMap[useInst]);
                }
            }
        }
        // Replace `cudaMalloc` with `cudaMallocAsync` and set stream
        if (instr == memObj->getMalloc()) {
            auto malloc = dyn_cast<CallInst>(instr);
            SmallVector<Value *> mallocAsyncArgs(malloc->arg_begin(), malloc->arg_end());
            // Handle cudaMalloc wrapper
            if (mallocAsyncArgs.size() == 1) {
                mallocAsyncArgs.push_back(memObj->getSize());
            }
            mallocAsyncArgs.push_back(loadStream);
            Type *voidPtrTy = PointerType::getUnqual(Type::getInt8PtrTy(M.getContext()));
            if (mallocAsyncArgs[0]->getType() != voidPtrTy) {
                // dbgs() << "Bitcast malloc argument" << *(mallocAsyncArgs[0]);
                AllocaInst *target = dyn_cast<AllocaInst>(mallocAsyncArgs[0]);
                mallocAsyncArgs[0] = BitCastInst::Create(Instruction::BitCast, mallocAsyncArgs[0], voidPtrTy, "", target->getNextNonDebugInstruction());
                // dbgs() << " to " << *(mallocAsyncArgs[0]) << "\n";
            }
            CallInst *mallocAsync = CallInst::Create(cudaMallocAsyncFunc, mallocAsyncArgs, "", insertPoint);
            malloc->replaceAllUsesWith(mallocAsync);
            memObj->setMalloc(mallocAsync);
            result.cudaCalls.erase(std::remove(result.cudaCalls.begin(), result.cudaCalls.end(), malloc), result.cudaCalls.end());
            result.cudaCalls.push_back(mallocAsync);
            malloc->eraseFromParent();
            // dbgs() << "Insert cudaMallocAsync: " << *mallocAsync << "\n";
        }
        else  {
            // dbgs() << "Move instruction " << *instr;
        }
    }
    // Update mallocRelatedInstr
    memObj->mallocRelatedInstrList = mallocRelatedInstrList;
    cloneMap.clear();

    SmallVector<Instruction *> memcpyRelatedInstrList;
    if (H2DMemNode->call) {
        // dbgs() << "All cudaMemcpyAsync (H2D) related instructions: \n";
        for (auto instr : H2DMemNode->memoryObj->getMemcpyRelatedInstrList(H2DMemNode->call)) {
            // dbgs() << *instr << "\n";
        }
        // dbgs() << "Moving cudaMemcpyAsync (H2D) related instructions before " << *(earliestNode->call) << "\n";
        for (auto originalInstr : H2DMemNode->memoryObj->getMemcpyRelatedInstrList(H2DMemNode->call)) {
            Instruction *instr = nullptr;
            if (originalInstr == H2DMemNode->call) {
                instr = originalInstr;
                instr->moveBefore(insertPoint);
            }
            else {
                instr = originalInstr->clone();
                cloneMap[originalInstr] = instr;
                instr->insertBefore(insertPoint);
                // dbgs() << "Clone instruction: " << *originalInstr << " to " << *instr << "\n";
            }
            memcpyRelatedInstrList.push_back(instr);
            // Replace uses of original instruction with cloned instruction
            for (auto &use : instr->operands()) {
                if (auto *useInst = dyn_cast<Instruction>(use)) {
                    if (cloneMap.find(useInst) != cloneMap.end()) {
                        instr->replaceUsesOfWith(useInst, cloneMap[useInst]);
                    }
                }
            }
            
            // dbgs() << "Before move: " << *instr << "\n";
            // Set stream for `cudaMemcpyAsync`
            if (instr == H2DMemNode->call) {
                loadStream = loadStreamBefore(H2DMemNode->stream, insertPoint, M.getContext());
                H2DMemNode->call->setArgOperand(4, loadStream);
                H2DMemNode->call->moveBefore(insertPoint);
            }
            // dbgs() << "Move instruction " << *instr << "\n";
        }
    }
    // Update memcpyRelatedInstr
    H2DMemNode->memoryObj->memcpyRelatedInstrMap[H2DMemNode->call] = memcpyRelatedInstrList;
    cloneMap.clear();

    // Add synchronization instructions after memory operations 
    // and before the nodes to be synchronized
    // dbgs() << "Adding synchronization instructions\n";
    insertPoint = memObj->getMalloc();
    if (H2DMemNode->call) {
        insertPoint = H2DMemNode->call;
    }
    for (auto node : H2DSyncNodeList) {
        // Dependancy H2DMemNode->node is already satisfied
        if (result.streamGraph.isDependent(H2DMemNode, node)) {
            continue;
        }
        // dbgs() << "Adding synchronization between " << *insertPoint << " and " << *(node->call) << "\n";
        assert(result.getKernelInvoke(node->call) && "KernelInvoke not found");
        auto dstInsertPt = result.getKernelInvoke(node->call)->getCudaPush();
        auto event = insertSyncBetween(insertPoint, H2DMemNode->stream, dstInsertPt, node->stream, M.getContext());
        StreamGraphEdge *edge = new StreamGraphEdge(event);
        // dbgs() << "CreateEdge: " << *insertPoint << " -> " << *(node->call) << "\n";
        edge->setEdge(H2DMemNode, node);
        result.streamGraph.addEdge(edge);
        result.cudaEventCreateCalls.push_back(edge->createCall);
    }

    // Remove redundant edges after moving H2DMemNode
    // dbgs() << "Deleting redundant edges\n";
    result.streamGraph.deleteAllRedundantEdges();
    // dbgs() << "Finish deleting redundant edges\n";
    // dbgs() << "Finish postponing memory operations of " << *(memObj->getMalloc()) << "\n";
    return true;
}


bool TaskAnalyzerPass::preponeMemoryOperations(MemoryObj *memObj, 
    SmallVector<StreamGraphNode *> invokeNodeList, Module &M) {
    StreamGraphNode *finalNode = nullptr;
    KernelInvoke *finalInvoke = nullptr;
    Instruction *insertPoint = nullptr;
    SmallVector<StreamGraphNode *> D2HsyncNodeList = invokeNodeList;
    CallInst *D2HMemcpyCall = nullptr;
    StreamGraphMemNode *D2HMemNode = nullptr;
    for (auto memcpy : memObj->getMemcpyList()) {
        cudaMemcpyKind kind = (cudaMemcpyKind)dyn_cast<ConstantInt>(memcpy->getArgOperand(3))->getZExtValue();
        if (kind == cudaMemcpyDeviceToHost) {
            D2HMemcpyCall = memcpy;
            D2HMemNode = dyn_cast<StreamGraphMemNode>(result.streamGraph.getNode(D2HMemcpyCall));
            break;
        }
    }
    // Sort invokeNodeList by the order in sequential graph
    std::sort(invokeNodeList.begin(), invokeNodeList.end(), 
        [&](StreamGraphNode *a, StreamGraphNode *b) {
            return result.streamGraph.getSequenceGraphIndex(a) < result.streamGraph.getSequenceGraphIndex(b);
        });

    if (invokeNodeList.size() == 0) {
        // dbgs() << "No invoke node found, postpone failed\n";
        assert(false && "No invoke node found, postpone failed");
    }
    finalNode = invokeNodeList[invokeNodeList.size() - 1];
    for (int j = 0; j < invokeNodeList.size() - 1; j ++) {
        if (!result.streamGraph.isDependent(invokeNodeList[j], finalNode)) {
            // if node i->j is satisfied, should not choose node i when preponing
            if (result.streamGraph.isDependent(finalNode, invokeNodeList[j])) {
                D2HsyncNodeList = invokeNodeList;
                assert(false && "Order in sequential graph and stream graph can not be satisfied simultaneously");
            }
            D2HsyncNodeList.push_back(invokeNodeList[j]);
        }
    }
    finalInvoke = result.getKernelInvoke(finalNode->call);
    // dbgs() << "Final node: " << *(finalNode->call) << "\n";
    if (D2HMemcpyCall) {
        // dbgs() << "cudaMemcpyAsync (D2H): " << *D2HMemcpyCall << "\n";
    }
    // dbgs() << "New sync list: \n";
    for (auto node : D2HsyncNodeList) {
        // dbgs() << *node->call << "\n";
    }
    // Move StreamGraphMemNode after the finalNode
    // dbgs() << "Insert point when preponing: " << *(finalInvoke->getInvoke()) << "\n";
    if (D2HMemNode) {
        D2HMemNode->setMemNodeKind(CUDA_MEMCPY_FREE);
        // dbgs() << "Edges of D2HMemNode before moving:\n";
        for (auto edge : D2HMemNode->outEdges) {
            if (edge->event) {
                // dbgs() << *(edge->event) << "\n";
            }
        }
        result.streamGraph.moveNodeAfter(D2HMemNode, finalNode);
        // dbgs() << "Edges of H2DMemNode:\n";
        for (auto edge : D2HMemNode->outEdges) {
            if (edge->event) {
                // dbgs() << *(edge->event) << "\n";
            }
        }
        // dbgs() << "Move node " << *(D2HMemNode->call) << " after " << *(finalNode->call) << "\n";
    }
    else {
        D2HMemNode = new StreamGraphMemNode(nullptr, finalNode->stream);
        D2HMemNode->setMemoryObj(memObj);
        D2HMemNode->setMemcpyKind(cudaMemcpyDeviceToHost);
        D2HMemNode->setMemNodeKind(CUDA_FREE);
        result.streamGraph.addNode(D2HMemNode);
        result.streamGraph.moveNodeAfter(D2HMemNode, finalNode);
        // dbgs() << "Move node " << *(D2HMemNode->memoryObj->malloc) << " after " << *(finalNode->call) << "\n";
    }

    // Move synchronization instructions after the earliest insert point
    insertPoint = finalInvoke->getInvoke()->getNextNonDebugInstruction();
    auto loadStream = loadStreamBefore(D2HMemNode->stream, insertPoint, M.getContext());
    // Add synchronization instructions after kernel invokes 
    // and before the memory related operations
    // dbgs() << "Adding synchronization instructions\n";
    for (auto node : D2HsyncNodeList) {
        // Dependancy node->D2HMemNode is already satisfied
        if (result.streamGraph.isDependent(node, D2HMemNode)) {
            continue;
        }
        auto srcInsertPt = node->call;
        auto event = insertSyncBetween(srcInsertPt, node->stream, insertPoint, D2HMemNode->stream, M.getContext());
        StreamGraphEdge *edge = new StreamGraphEdge(event);
        // dbgs() << "Event: " << *event << "\n";
        edge->setEdge(node, D2HMemNode);
        // dbgs() << "SetEdge: " << *(node->call) << " -> " << *insertPoint << "\n";
        result.streamGraph.addEdge(edge);
        result.cudaEventCreateCalls.push_back(edge->createCall);
    }
    // Move instructions of node after the finalNode
    std::map<Instruction *, Instruction *> cloneMap;
    SmallVector<Instruction *> memcpyRelatedInstrList;
    if (D2HMemNode->call) {
        // dbgs() << "All cudaMemcpyAsync (D2H) related instructions: \n";
        for (auto instr : D2HMemNode->memoryObj->getMemcpyRelatedInstrList(D2HMemNode->call)) {
            // dbgs() << *instr << "\n";
        }
        // dbgs() << "Moving cudaMemcpyAsync (D2H) related instructions after " << *(finalNode->call) << "\n";

        for (auto originalInstr : D2HMemNode->memoryObj->getMemcpyRelatedInstrList(D2HMemNode->call)) {
            // Copy memcpy-related instruction after insert point
            Instruction *instr = nullptr;
            if (originalInstr == D2HMemNode->call) {
                instr = originalInstr;
                instr->moveBefore(insertPoint);
            }
            else {
                instr = originalInstr->clone();
                cloneMap[originalInstr] = instr;
                instr->insertBefore(insertPoint);
                // dbgs() << "D2HMemNode->call: " << *D2HMemNode->call << "\n";
                // dbgs() << "Clone instruction: " << *originalInstr << " to " << *instr << "\n";
            }
            memcpyRelatedInstrList.push_back(instr);
            // Replace uses of original instruction with cloned instruction
            for (auto &use : instr->operands()) {
                if (auto *useInst = dyn_cast<Instruction>(use)) {
                    if (cloneMap.find(useInst) != cloneMap.end()) {
                        instr->replaceUsesOfWith(useInst, cloneMap[useInst]);
                    }
                }
            }

            // dbgs() << "Before move: " << *instr << "\n";
            // Set stream for `cudaMemcpyAsync`
            if (instr == D2HMemNode->call) {
                auto originalStream = dyn_cast<Instruction>(D2HMemNode->call->getArgOperand(4));
                assert(originalStream && "Stream for cudaMemcpyAsync is not found");
                loadStream = loadStreamBefore(D2HMemNode->stream, insertPoint, M.getContext());
                D2HMemNode->call->setArgOperand(4, loadStream);
                // Remember move instruction before insert point after loading stream
                D2HMemNode->call->moveBefore(insertPoint);
            }
            // dbgs() << "Move instruction " << *instr << "\n";
        }
    }
    // Update memcpyRelatedInstr
    D2HMemNode->memoryObj->memcpyRelatedInstrMap[D2HMemNode->call] = memcpyRelatedInstrList;
    cloneMap.clear();

    // Moving cudaFree-related instructions of node after the finalNode
    // dbgs() << "Moving cudaFree related instructions after " << *(finalNode->call) << "\n";
    DT.recalculate(*(finalInvoke->getInvoke()->getFunction()));
    SmallVector<Instruction *> freeRelatedInstrList;
    for (auto originalInstr : memObj->getFreeRelatedInstrList()) {
        Instruction *instr = nullptr;
        if (originalInstr == memObj->getFree()) {
            instr = originalInstr;
            instr->moveBefore(insertPoint);
        }
        else {
            instr = originalInstr->clone();
            cloneMap[originalInstr] = instr;
            instr->insertBefore(insertPoint);
            // dbgs() << "Clone instruction: " << *originalInstr << " to " << *instr << "\n";
        }
        freeRelatedInstrList.push_back(instr);
        
        // Replace uses of original instruction with cloned instruction
        for (auto &use : instr->operands()) {
            if (auto *useInst = dyn_cast<Instruction>(use)) {
                if (cloneMap.find(useInst) != cloneMap.end()) {
                    instr->replaceUsesOfWith(useInst, cloneMap[useInst]);
                }
            }
        }

        // Replace `cudaFree` with `cudaFreeAsync` and set stream
        if (instr == memObj->getFree()) {
            auto free = dyn_cast<CallInst>(instr);
            SmallVector<Value *> freeAsyncArgs(free->arg_begin(), free->arg_end());
            freeAsyncArgs.push_back(loadStream);

            CallInst *freeAsync = CallInst::Create(cudaFreeAsyncFunc, freeAsyncArgs, "", insertPoint);
            free->replaceAllUsesWith(freeAsync);
            memObj->setFree(freeAsync);
            result.cudaCalls.erase(std::remove(result.cudaCalls.begin(), result.cudaCalls.end(), free), result.cudaCalls.end());
            result.cudaCalls.push_back(freeAsync);
            free->eraseFromParent();
            // dbgs() << "Insert cudaFreeAsync: " << *freeAsync << "\n";
        }
        else  {
            // dbgs() << "Move instruction " << *instr << "\n";
        }
    }
    // Update freeRelatedInstr
    memObj->freeRelatedInstrList = freeRelatedInstrList;

    // Remove redundant edges after moving D2HMemNode
    result.streamGraph.deleteAllRedundantEdges();
    // dbgs() << "Finish preponing memory operations of " << *(memObj->getFree()) << "\n\n";
    return true;
}

// Schedule memory operations in stream graph. This function attempt to shorten the lifetime of memory objects
// by postponing cudaMalloc, cudaMemcpy and preponing cudaFree to the earliest possible point.
bool TaskAnalyzerPass::scheduleMemoryOperations(Module &M) {
    // dbgs() << "TaskAnalyzerPass: scheduleMemoryOperations\n";
    // Schedule for each memory object
    for (auto memObj : result.memObjList) {
        SmallVector<StreamGraphNode *> invokeNodeList;
        for (auto invokeInfo : memObj->getKernelInvokeList()) {
            StreamGraphNode *node = result.streamGraph.getNode(invokeInfo->getInvoke());
            if (node == nullptr) {
                // dbgs() << "Kernel invoke node not found in stream graph\n";
                return true;
            }
            invokeNodeList.push_back(node);
        }

        // dbgs() << "Scheduling memory object: " << *memObj->getMalloc() << "\n";
        postponeMemoryOperations(memObj, invokeNodeList, M);
        preponeMemoryOperations(memObj, invokeNodeList, M);
    }
    return true;
}

// Find an insert point which dominates all CUDA function calls 
// and postdominates all malloc sizes, and insert memory requirement calculation.
// Assuming that all kzernel invokes are in the same function.
bool TaskAnalyzerPass::insertMemoryRequirement(Module &M) {
    // dbgs() << "TaskAnalyzerPass: insertMemoryRequirement\n";
    assert(result.kernelInvokes.size() > 0 && "No kernel invoke found");
    Function *F = result.kernelInvokes[0]->getFunction();
    for (auto *CI : result.kernelInvokes) {
        assert(CI->getFunction() == F && "Kernel invokes are not in the same function");
    }

    Instruction *insertPoint = nullptr;
    SmallVector<CallInst *> cudaMallocCalls;
    // Find an insert point which postdominates all malloc size and dominates all CUDA function calls
    PDT.recalculate(*F);
    DT.recalculate(*F);
    for (auto memObj : result.memObjList) {
        // dbgs() << "Malloc: " << *(memObj->getMalloc()) << ", size: " << *memObj->getSize() << "\n";
        cudaMallocCalls.push_back(memObj->getMalloc());
    }
    // dbgs() << "Begin to find the insert point\n";
    // Find an insert point which postdominates all malloc size and dominates all CUDA function calls except cudaMalloc

    for (auto inst: result.cudaCalls) {
        auto name = getDemangledName(inst->getCalledFunction()->getName().str());
        if (name == "cudaStreamCreate" || name == "cudaEventCreate") {
            continue;
        }
        if (insertPoint != nullptr) {
            break;
        }
        // dbgs() << "Instruction: " << *inst << "\n";
        bool flag = true;
        for (auto memObj : result.memObjList) {
            // Insert point should postdominate all arguments of cudaMalloc
            assert(memObj->getMalloc() && "Malloc is not found");
            Instruction *mallocPtr = dyn_cast<Instruction>(memObj->getMalloc()->getArgOperand(0));
            assert(mallocPtr && "Malloc pointer is not found");
            if (!PDT.dominates(inst, mallocPtr)) {
                flag = false;
                break;
            }
            if (auto sizeInstruction = dyn_cast<Instruction>(memObj->getSize())) {
                if (!PDT.dominates(inst, sizeInstruction)) {
                    flag = false;
                    break;
                }
            }
        }
        if (!flag) {
            continue;
        }
        for (auto *CI : result.cudaCalls) {
            // dbgs() << "CUDA call: " << *CI << "\n";
            auto name = getDemangledName(CI->getCalledFunction()->getName().str());
            if (name == "cudaStreamCreate" || name == "cudaEventCreate") {
                continue;
            }
            if (!DT.dominates(inst, CI) && inst != CI) {
                // dbgs() << *inst << " does not dominate " << *CI << "\n";
                flag = false;
                break;
            }
        }
        if (!flag) {
            continue;
        }
        insertPoint = inst;
    }

    // dbgs() << "Exit finding insert point\n";

    // Use dynamic resource requirement detection when static assertion failed
    if (insertPoint == nullptr) {
        return false;
    }
    // dbgs() << "Insert point: " << *insertPoint << "\n";

    LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>(*(insertPoint->getFunction())).getLoopInfo();
    
    bool isInLoop = (LI.getLoopFor(insertPoint->getParent()) != nullptr);
    if (isInLoop) {
        bool isInLoopBody = true;
        if (insertPoint->getParent() == LI.getLoopFor(insertPoint->getParent())->getLoopPreheader()) {
            isInLoopBody = false;
        }
        else if (insertPoint->getParent() == LI.getLoopFor(insertPoint->getParent())->getExitBlock()) {
            isInLoopBody = false;
        }
        if (isInLoopBody) {
            // dbgs() << "Insert point is in loop\n";
            while (Loop *ParentLoop = LI.getLoopFor(insertPoint->getParent())) {
                BasicBlock *preheader = ParentLoop->getLoopPreheader();
                if (!preheader) {
                    BasicBlock *header = ParentLoop->getHeader();
                    BasicBlock *latch = ParentLoop->getLoopLatch();
                    for (auto *Pred : predecessors(header)) {
                        if (!ParentLoop->contains(Pred)) {
                            preheader = Pred;
                            break;
                        }
                    }
                    if (!preheader) {
                        // dbgs() << "No preheader found, using header's predecessor\n";
                        preheader = header->getSinglePredecessor();
                    }
                }
                // dbgs() << "Found preheader: " << *preheader << "\n";
                insertPoint = preheader->getTerminator()->getPrevNonDebugInstruction();
                // dbgs() << "New insert point: " << *insertPoint << "\n";
            }
        }
    }

    // Insert memory requirement calculation
    IRBuilder<> builder(insertPoint);
    Value *totalMemSize = nullptr;
    if (useMemorySchedule) {
        // Build memory graph and calculate memory consumption
        MemoryGraph memoryGraph = result.streamGraph.toMemoryGraph();
        // memoryGraph.generateMemoryGraphDotFile();
        SmallVector<Value *, 3> memoryGraphLLVMArrays = memoryGraph.toLLVMArray(insertPoint);
        totalMemSize = builder.CreateCall(cudaPredictPeakMemory, memoryGraphLLVMArrays, "totalMemSize");
        // dbgs() << "Optimized memory size: " << *totalMemSize << "\n";
    }
    else {
        for (auto memObj : result.memObjList) {
            if (totalMemSize == nullptr) {
                totalMemSize = memObj->getSize();
            }
            else {
                totalMemSize = builder.CreateAdd(totalMemSize, memObj->getSize(), "task_mem");
            }
        }
        // dbgs() << "originalMaxMemory: " << *totalMemSize << "\n";
        // dbgs() << "Memory size without memory schedule: " << *totalMemSize << "\n";
    }

    // // Insert thread size calculation
    Value *gridDim = nullptr;
    Value *blockDim = nullptr;
    LLVMContext &context = M.getContext();
    // Move cudaStreamCreate after insertPoint
    SmallVector<CallInst *> toEraseInstr;
    DT.recalculate(*F);
    for (auto CI = result.cudaStreamCreateCalls.rbegin(); CI != result.cudaStreamCreateCalls.rend(); CI++) {
        // Erase unused cudaStreamCreate
        if ((*CI)->getOperand(0)->getNumUses() == 1) {
            assert(isa<GlobalVariable>((*CI)->getOperand(0)));
            toEraseInstr.push_back(*CI);
            continue;
        }
    }
    for (auto CI : toEraseInstr) {
        // dbgs() << "Erase unused " << *CI << "\n";
        auto GV = dyn_cast<GlobalVariable>(CI->getOperand(0));
        CI->eraseFromParent();
        GV->eraseFromParent();
        result.cudaStreamCreateCalls.erase(std::remove(result.cudaStreamCreateCalls.begin(), result.cudaStreamCreateCalls.end(), CI), result.cudaStreamCreateCalls.end());
    }

    StreamGraph &graph = result.streamGraph;
    Value *gridSize = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
    Value *blockSize = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
    Value *numThread = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
    Value *numRegister = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
    Value *numSharedMem = ConstantInt::get(Type::getInt32Ty(M.getContext()), 0);
    // Select the first kernel invoke of each stream
    for (int i = 0; i < graph.getNumStream(); i ++) {
        SmallVector<StreamGraphNode *> streamVector = graph.getStreamVector(i);
        for (int j = 0; j < streamVector.size(); j ++) {
            if (streamVector[j]->kind == KERNEL_INVOKE) {
                // dbgs() << "first kernel invoke of stream " << i << ": " << *(streamVector[j]->call) << "\n";
                auto kernelInvoke = result.getKernelInvoke(streamVector[j]->call);
                if (DT.dominates(insertPoint, kernelInvoke->gridDim.getCtor())) {
                    kernelInvoke->gridDim.moveCtorBefore(insertPoint);
                }
                if (DT.dominates(insertPoint, kernelInvoke->blockDim.getCtor())) {
                    kernelInvoke->blockDim.moveCtorBefore(insertPoint);
                }
                Value *curGridSize = kernelInvoke->gridDim.getTotalSize();
                Value *curBlockSize = kernelInvoke->blockDim.getTotalSize();
                Value *curNumThread = builder.CreateMul(curGridSize, curBlockSize, "numThread");
                Value *curNumRegister = builder.CreateMul(curNumThread, ConstantInt::get(Type::getInt32Ty(M.getContext()), kernelInvoke->numRegister), "numRegister");
                Value *curNumSharedMem = builder.CreateMul(curGridSize, ConstantInt::get(Type::getInt32Ty(M.getContext()), kernelInvoke->numSharedMem), "numSharedMem");
                numThread = builder.CreateAdd(numThread, curNumThread, "numThread");
                numRegister = builder.CreateAdd(numRegister, curNumRegister, "numRegister");
                numSharedMem = builder.CreateAdd(numSharedMem, curNumSharedMem, "numSharedMem");
                break;
            }
        }
    }
    // dbgs() << "number of threads: " << *numThread << ", number of registers: " << *numRegister << ", number of shared memory: " << *numSharedMem << "\n";

    // Insert cudaTaskSchedule
    ConstantInt *numStream = ConstantInt::get(Type::getInt32Ty(M.getContext()), result.streamGraph.getNumStream());
    ConstantInt *useMemoryScheduleVal = ConstantInt::get(Type::getInt8Ty(M.getContext()), useMemorySchedule);
    SmallVector<Value *> args{numThread, numRegister, numSharedMem, totalMemSize, numStream, useMemoryScheduleVal};
    CallInst *newCall = builder.CreateCall(cudaTaskSchedule, args);
    // dbgs() << "number of stream: " << *numStream << "\n";
    // dbgs() << "cudaTaskSchedule type: " << *cudaTaskSchedule.getFunctionType() << "\n";
    // dbgs() << "insertPoint: " << *insertPoint << "\n";
    // dbgs() << "cudaTaskSchedule inserted: " << *newCall << "\n";
    // dbgs() << "Use memory schedule: " << useMemorySchedule << "\n";
    
    // Move all 'cudaStreamCreate' and 'cudaEventCreate' after 'cudaTaskSchedule'
    for (auto CI = result.cudaStreamCreateCalls.rbegin(); CI != result.cudaStreamCreateCalls.rend(); CI++) {
        if (DT.dominates(newCall, *CI)) {
            continue;
        }
        (*CI)->moveAfter(newCall);
    }
    for (auto CI : result.cudaEventCreateCalls) {
        if (!DT.dominates(newCall, CI)) {
            CI->moveAfter(newCall);
            // dbgs() << "Move " << *CI << " after insertPoint: " << *newCall << "\n";
        }
    }
    insertPoint->getParent()->print(dbgs());

    return true;
}

// Find an insert point which dominates all kernel invokes 
// and postdominates all malloc sizes, and insert memory requirement calculation
// Assert all kernel invokes are in the same function
bool TaskAnalyzerPass::insertLazySchedule(Module &M) {
    // dbgs() << "TaskAnalyzerPass: insertLazySchedule\n";
    assert(result.kernelInvokes.size() > 0 && "No kernel invoke found");
    Function *F = result.kernelInvokes[0]->getFunction();
    for (auto *CI : result.kernelInvokes) {
        // Kernel invokes are not in the same function
        if (CI->getFunction() != F) {
            return false;
        }
    }

    Instruction *insertPoint = nullptr;
    // Find an insert point which postdominates all malloc size and dominates all kernel invokes
    PDT.recalculate(*F);
    DT.recalculate(*F);
    for (auto &bb : F->getBasicBlockList()) {
        for (auto &inst : bb) {
            if (insertPoint != nullptr) {
                break;
            }
            bool flag = true;
            for (auto memObj : result.memObjList) {
                if (auto mallocInst = memObj->getMalloc()) {
                    if (!PDT.dominates(&inst, mallocInst)) {
                        flag = false;
                        break;
                    }
                }
            }
            if (!flag) {
                continue;
            }
            for (auto *CI : result.kernelInvokes) {
                if (!DT.dominates(&inst, CI)) {
                    flag = false;
                    break;
                }
            }
            if (!flag) {
                continue;
            }
            insertPoint = &inst;
        }
    }

    if (insertPoint == nullptr) {
        return false;
    }
    else if (auto mallocInst = dyn_cast<CallInst>(insertPoint)) {
        // use next instruction as insert point if it is cudaMalloc
        if (mallocInst->getCalledFunction()->getName() == "cudaMalloc") {
            insertPoint = insertPoint->getNextNonDebugInstruction();
        }
    }
    // dbgs() << "Insert point: " << *insertPoint << "\n";
    // Insert cudaTaskSchedule
    IRBuilder<> builder(insertPoint);
    CallInst *newCall = builder.CreateCall(cudaTaskScheduleLazy, {});
    // dbgs() << "cudaTaskScheduleLazy inserted: " << *newCall << "\n";
    return true;
}

// Insert cudaLaunchKernelPrepare before each `__cudaPushCallConfiguration`
void TaskAnalyzerPass::insertCudaLaunchKernelPrepare(Module &M) {
    LLVMContext &context = M.getContext();
    IRBuilder<> builder(context);
    for (auto &F : M) {
        if (F.isDeclaration()) {
            continue;
        }
        for (auto &BB : F) {
            for (auto &I : BB) {
                if (auto *CI = dyn_cast<CallInst>(&I)) {
                    // Skip indirect calls
                    if (CI->getCalledFunction() == nullptr) {
                        continue;
                    }
                    if (CI->getCalledFunction()->getName().contains("__cudaPushCallConfiguration")) {
                        SmallVector<Value *, 4> args;
                        for (int i = 0; i < 4; i ++) {
                            args.push_back(CI->getArgOperand(i));
                        }
                        builder.SetInsertPoint(CI);
                        CallInst *call = builder.CreateCall(cudaLaunchKernelPrepare, args);
                        // dbgs() << "cudaLaunchKernelPrepare inserted before: " << *CI << "\n";
                        // dbgs() << "cudaLaunchKernelPrepare: " << *call << "\n";
                    }
                }
            }
        }
    }
}

// Insert fakeAddrLookup for all pointer arguments of CUDA calls after lazy schedule
void TaskAnalyzerPass::insertFakeAddrLookup(Module &M)  {
    // dbgs() << "TaskAnalyzerPass: insertFakeAddrLookup\n";
    SmallVector<Instruction *> toErase;
    IRBuilder<> builder(M.getContext());
    std::set<Value *> visited;
    SmallVector<CallInst *> potentialLazyCalls(result.cudaCalls.begin(), result.cudaCalls.end());
    potentialLazyCalls.append(result.kernelInvokes.begin(), result.kernelInvokes.end());
    // @TODO: make sure all potential lazy calls are considered
    for (auto *CI : potentialLazyCalls) {
        // Skip cudaMalloc calls which would not contain a fake address
        if (CI->getCalledFunction()->getName().contains("cudaMalloc")) {
            continue;
        }
        builder.SetInsertPoint(CI);
        // Insert fakeAddrLookup for all pointer arguments
        for (auto &arg : CI->args()) {
            if (visited.count(arg)) {
                continue;
            }
            // @TODO: be compatible with the opaque pointer feature in LLVM 17
            // Insert bitcast for all non-void pointers
            if (arg->getType()->isPointerTy() && isa<Instruction>(arg)) {
                Value *ptr = arg.get();
                bool isVoidPtr = ptr->getType()->getPointerElementType()->isVoidTy();
                if (!isVoidPtr) {
                    ptr = builder.CreateBitCast(ptr, Type::getInt8PtrTy(M.getContext()));
                }
                Value *newCall = builder.CreateCall(fakeAddrLookup, {ptr});
                // // dbgs() << "fakeAddrLookup inserted: " << *newCall << " for " << *arg << "\n";
                if (!isVoidPtr) {
                    newCall = builder.CreateBitCast(newCall, arg->getType());
                }
                for (auto user : arg->users()) {
                    if (auto *userInst = dyn_cast<Instruction>(user)) {
                        if (userInst == newCall || userInst == ptr) {
                            continue;
                        }
                        userInst->replaceUsesOfWith(arg, newCall);
                    }
                }
                visited.insert(arg);
            }
        }
    }
    return;
}

// Insert `void cudaInitialize(bool is_lazy)` to the entry of main function
void TaskAnalyzerPass::insertCudaInitialize(Module &M) {
    LLVMContext &context = M.getContext();
    IRBuilder<> builder(context);
    Function *mainFunc = M.getFunction("main");
    if (mainFunc == nullptr) {
        // dbgs() << "main function not found in file " << M.getSourceFileName() << "\n";
        return;
    }
    BasicBlock &entry = mainFunc->getEntryBlock();
    builder.SetInsertPoint(&entry, entry.getFirstInsertionPt());
    Type *boolTy = Type::getInt1Ty(context);
    SmallVector<Type*, 1> cudaInitializeArgsType = {boolTy};
    FunctionType *cudaInitializeTy = FunctionType::get(Type::getVoidTy(context), cudaInitializeArgsType, false);
    SmallVector<Value*, 1> cudaInitializeArgs = {ConstantInt::get(boolTy, result.isLazy)};
    builder.CreateCall(M.getOrInsertFunction("cudaInitialize", cudaInitializeTy), cudaInitializeArgs);
}

void TaskAnalyzerPass::insertCudaFinalize(Module &M) {
    LLVMContext &context = M.getContext();
    IRBuilder<> builder(context);
    Function *mainFunc = M.getFunction("main");
    if (mainFunc == nullptr) {
        // dbgs() << "main function not found\n";
        return;
    }
    for (auto &BB : *mainFunc) {
        if (ReturnInst *retInst = dyn_cast<ReturnInst>(BB.getTerminator())) {
            builder.SetInsertPoint(retInst);
            builder.CreateCall(M.getOrInsertFunction("cudaFinalize", Type::getVoidTy(context)));
        }
    }
}

bool TaskAnalyzerPass::removeDeadCode(Module &M) {
    bool changed = false;
    // Remove dead code
    for (auto &F : M) {
        auto &bbList = F.getBasicBlockList();
        SmallVector<Instruction *> toRemoveInstr;
        for (auto BB = bbList.rbegin(); BB != bbList.rend(); ++BB) {
            for (auto it = (*BB).rbegin(); it != (*BB).rend(); ++it) {
                Instruction *I = &*it;
                if (isInstructionTriviallyDead(I)) {
                    toRemoveInstr.push_back(I);
                }
            }
        }
        for (auto I : toRemoveInstr) {
            // dbgs() << "Remove dead code: " << *I << "\n";
            I->eraseFromParent();
            changed = true;
        }
    }

    // Remove unused cudaStreamCreate
    SmallVector<CallInst *> toEraseInstr;
    for (auto CI = result.cudaStreamCreateCalls.rbegin(); CI != result.cudaStreamCreateCalls.rend(); CI++) {
        // Erase unused cudaStreamCreate
        if ((*CI)->getOperand(0)->getNumUses() == 1) {
            assert(isa<GlobalVariable>((*CI)->getOperand(0)));
            toEraseInstr.push_back(*CI);
            continue;
        }
    }
    for (auto CI : toEraseInstr) {
        // dbgs() << "Erase unused stream" << *CI << "\n";
        auto GV = dyn_cast<GlobalVariable>(CI->getOperand(0));
        CI->eraseFromParent();
        GV->eraseFromParent();
        result.cudaStreamCreateCalls.erase(std::remove(result.cudaStreamCreateCalls.begin(), result.cudaStreamCreateCalls.end(), CI), result.cudaStreamCreateCalls.end());
        changed = true;
    }

    return changed;
}

bool TaskAnalyzerPass::isDim3Struct(Type *Ty) {
  if (isa<PointerType>(Ty))
    Ty = dyn_cast<PointerType>(Ty)->getPointerElementType();
  if (!isa<StructType>(Ty) || dyn_cast<StructType>(Ty)->isLiteral())
    return false;
  return dyn_cast<StructType>(Ty)->getStructName() == "struct.dim3";
}

char TaskAnalyzerPass::ID = 0;
