#include "passes/wrapper/wrapper_pass.h"
#include "passes/analyzer/analyzer.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Pass.h"


using namespace llvm;

WrapperPass::WrapperPass(): ModulePass(ID) {}

WrapperPass::~WrapperPass() {}

void WrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
    // AU.addRequired<TaskAnalyzerPass>();
}

// CUDA related functions Wrappers
void WrapperPass::createCudaMallocWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context)->getPointerTo();
    Type *sizeTy = Type::getInt64Ty(context);
    SmallVector<Type*, 2> cudaMallocArgs = {voidPtrTy, sizeTy};
    FunctionType *cudaMallocTy = FunctionType::get(retTy, cudaMallocArgs, false);
    cudaMallocWrapper = M.getOrInsertFunction("cudaMallocWrapper", cudaMallocTy);
}

void WrapperPass::createCudaMallocAsyncWrapper(Module &M) {
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
    cudaMallocAsyncWrapper = M.getOrInsertFunction("cudaMallocAsyncWrapper", cudaMallocAsyncTy);
}

void WrapperPass::createCudaStreamCreateWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);
    Type *sizeTy = Type::getInt64Ty(context);
    Type *intTy = Type::getInt32Ty(context);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);
    streamStructPtTy = PointerType::getUnqual(streamStructPtTy);

    SmallVector<Type*, 1> cudaStreamCreateArgs = {streamStructPtTy};
    FunctionType *cudaStreamCreateTy = FunctionType::get(retTy, cudaStreamCreateArgs, false);
    cudaStreamCreateWrapper = M.getOrInsertFunction("cudaStreamCreateWrapper", cudaStreamCreateTy);
}

void WrapperPass::createCudaMemcpyWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);
    Type *sizeTy = Type::getInt64Ty(context);
    Type *intTy = Type::getInt32Ty(context);
    SmallVector<Type*, 4> cudaMemcpyArgs = {voidPtrTy, voidPtrTy, sizeTy, intTy};
    FunctionType *cudaMemcpyTy = FunctionType::get(retTy, cudaMemcpyArgs, false);
    cudaMemcpyWrapper = M.getOrInsertFunction("cudaMemcpyWrapper", cudaMemcpyTy);
}

void WrapperPass::createCudaMemcpyAsyncWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);
    Type *sizeTy = Type::getInt64Ty(context);
    Type *intTy = Type::getInt32Ty(context);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);

    SmallVector<Type*, 5> cudaMemcpyAsyncArgs = {voidPtrTy, voidPtrTy, sizeTy, intTy, streamStructPtTy};
    FunctionType *cudaMemcpyAsyncTy = FunctionType::get(retTy, cudaMemcpyAsyncArgs, false);
    cudaMemcpyAsyncWrapper = M.getOrInsertFunction("cudaMemcpyAsyncWrapper", cudaMemcpyAsyncTy);
}

void WrapperPass::createCudaMemsetWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);
    Type *intTy = Type::getInt32Ty(context);
    Type *sizeTy = Type::getInt64Ty(context);
    SmallVector<Type*, 4> cudaMemsetArgs = {voidPtrTy, intTy, sizeTy};
    FunctionType *cudaMemsetTy = FunctionType::get(retTy, cudaMemsetArgs, false);
    cudaMemsetWrapper = M.getOrInsertFunction("cudaMemsetWrapper", cudaMemsetTy);
}

void WrapperPass::createCudaFreeWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);
    SmallVector<Type*, 1> cudaFreeArgs = {voidPtrTy};
    FunctionType *cudaFreeTy = FunctionType::get(retTy, cudaFreeArgs, false);
    cudaFreeWrapper = M.getOrInsertFunction("cudaFreeWrapper", cudaFreeTy);
}

void WrapperPass::createCudaFreeAsyncWrapper(Module &M) {
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
    cudaFreeAsyncWrapper = M.getOrInsertFunction("cudaFreeAsyncWrapper", cudaFreeAsyncTy);
}

void WrapperPass::createCudaLaunchKernelWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *KernelFuncPtrTy = PointerType::getUnqual(Type::getInt8Ty(context));
    Type *dim3Int64Ty = Type::getInt64Ty(context);
    Type *dim3Int32Ty = Type::getInt32Ty(context);
    Type *argTy = PointerType::getUnqual(PointerType::getUnqual(Type::getInt8Ty(context)));
    Type *shmSizeTy = Type::getInt64Ty(context);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    
    // Type *streamTy = PointerType::getUnqual(StructType::getTypeByName(context, "struct.CUstream_st"));
    Type *streamTy = PointerType::getUnqual(streamStructTy);
    SmallVector<Type*, 8> cudaLaunchArgs = {KernelFuncPtrTy, dim3Int64Ty, dim3Int32Ty, dim3Int64Ty, dim3Int32Ty, argTy, shmSizeTy, streamTy};
    FunctionType *cudaLaunchTy = FunctionType::get(retTy, cudaLaunchArgs, false);
    cudaLaunchKernelWrapper = M.getOrInsertFunction("cudaLaunchKernelWrapper", cudaLaunchTy);
}

void WrapperPass::createCudaEventCreateWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    StructType *eventStructTy = StructType::getTypeByName(context, "struct.CUevent_st");
    // struct.CUstream_st is not defined in the program
    if (eventStructTy == nullptr) {
        eventStructTy = StructType::create(context, "struct.CUevent_st");
    }
    Type *eventStructPtTy = PointerType::getUnqual(eventStructTy);
    eventStructPtTy = PointerType::getUnqual(eventStructPtTy);

    SmallVector<Type*, 1> cudaEventCreateArgs = {eventStructPtTy};
    FunctionType *cudaEventCreateTy = FunctionType::get(retTy, cudaEventCreateArgs, false);
    cudaEventCreateWrapper = M.getOrInsertFunction("cudaEventCreateWrapper", cudaEventCreateTy);
}

void WrapperPass::createCudaEventRecordWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);

    StructType *eventStructTy = StructType::getTypeByName(context, "struct.CUevent_st");
    // struct.CUstream_st is not defined in the program
    if (eventStructTy == nullptr) {
        eventStructTy = StructType::create(context, "struct.CUevent_st");
    }
    Type *eventStructPtTy = PointerType::getUnqual(eventStructTy);

    SmallVector<Type*, 2> cudaEventRecordArgs = {eventStructPtTy, streamStructPtTy};
    FunctionType *cudaEventRecordTy = FunctionType::get(retTy, cudaEventRecordArgs, false);
    cudaEventRecordWrapper = M.getOrInsertFunction("cudaEventRecordWrapper", cudaEventRecordTy);
}

void WrapperPass::createCudaStreamWaitEventWrapper(Module &M) {
    LLVMContext &context = M.getContext();
    Type *retTy = Type::getInt32Ty(context);
    Type *flagTy = Type::getInt32Ty(context);
    Type *voidPtrTy = Type::getInt8PtrTy(context);
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);

    StructType *eventStructTy = StructType::getTypeByName(context, "struct.CUevent_st");
    // struct.CUstream_st is not defined in the program
    if (eventStructTy == nullptr) {
        eventStructTy = StructType::create(context, "struct.CUevent_st");
    }
    Type *eventStructPtTy = PointerType::getUnqual(eventStructTy);

    SmallVector<Type*, 2> cudaStreamWaitEventArgs = {streamStructPtTy, eventStructPtTy, flagTy};
    FunctionType *cudaStreamWaitEventTy = FunctionType::get(retTy, cudaStreamWaitEventArgs, false);
    cudaStreamWaitEventWrapper = M.getOrInsertFunction("cudaStreamWaitEventWrapper", cudaStreamWaitEventTy);
}

void WrapperPass::replaceCudaWrapper(IRBuilder<> &builder, CallInst *callInst, FunctionCallee &callee) {
    if (callee.getCallee() == nullptr) {
        // dbgs() << "callee not found\n";
        exit(1);
    }

    // dbgs() << "found call: " << *callInst << "\n";
    builder.SetInsertPoint(callInst);
    SmallVector<Value *> args(callInst->arg_begin(), callInst->arg_end());
    CallInst *newCall = builder.CreateCall(callee, args);
    callInst->replaceAllUsesWith(newCall);
    // dbgs() << "replaced with: " << *newCall << "\n";
}

bool WrapperPass::runOnModule(Module &M) {
#ifdef NO_WRAPPER
    return true;
#endif
    initialization(M);
    for (Function &F : M) {
        for (BasicBlock &BB : F) {
            if (F.isDeclaration()) continue;

            IRBuilder<> builder(BB.getContext());
            SmallVector<Instruction *> toErase;
            // find CUDA functions and replace them with wrapper
            for (Instruction &I : BB) {
                if (CallInst *callInst = dyn_cast<CallInst>(&I)) {
                    Function *calledFunction = callInst->getCalledFunction();
                    if (calledFunction == nullptr) continue;
                    StringRef funcName = calledFunction->getName();

                    if (funcName == "cudaMalloc") {
                        replaceCudaWrapper(builder, callInst, cudaMallocWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaMallocAsync") {
                        replaceCudaWrapper(builder, callInst, cudaMallocAsyncWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaStreamCreate") {
                        replaceCudaWrapper(builder, callInst, cudaStreamCreateWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaMemcpy") {
                        replaceCudaWrapper(builder, callInst, cudaMemcpyWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaMemcpyAsync") {
                        replaceCudaWrapper(builder, callInst, cudaMemcpyAsyncWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaMemset") {
                        replaceCudaWrapper(builder, callInst, cudaMemsetWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaFree") {
                        replaceCudaWrapper(builder, callInst, cudaFreeWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaFreeAsync") {
                        replaceCudaWrapper(builder, callInst, cudaFreeAsyncWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaLaunchKernel") {
                        replaceCudaWrapper(builder, callInst, cudaLaunchKernelWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaEventCreate") {
                        replaceCudaWrapper(builder, callInst, cudaEventCreateWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaEventRecord") {
                        replaceCudaWrapper(builder, callInst, cudaEventRecordWrapper);
                        toErase.push_back(callInst);
                    }
                    else if (funcName == "cudaStreamWaitEvent") {
                        replaceCudaWrapper(builder, callInst, cudaStreamWaitEventWrapper);
                        toErase.push_back(callInst);
                    }
                }
            }

            // erase replaced functions
            if (toErase.size() > 0) {
                for (Instruction *inst : toErase) {
                    inst->eraseFromParent();
                }
            }
        }
    }
    // dbgs() << "Finish replacing CUDA functions with wrappers\n";
    std::error_code EC;
    raw_fd_ostream dest("tmp_opt.ll", EC);
    M.print(dest, nullptr);
    return true;
}

void WrapperPass::initialization(Module &M) {
    createCudaMallocWrapper(M);
    createCudaMallocAsyncWrapper(M);
    createCudaStreamCreateWrapper(M);
    createCudaMemcpyWrapper(M);
    createCudaMemcpyAsyncWrapper(M);
    createCudaMemsetWrapper(M);
    createCudaFreeWrapper(M);
    createCudaFreeAsyncWrapper(M);
    createCudaLaunchKernelWrapper(M);
    createCudaEventCreateWrapper(M);
    createCudaEventRecordWrapper(M);
    createCudaStreamWaitEventWrapper(M);
}


char WrapperPass::ID = 0;

// Register for opt
static RegisterPass<WrapperPass> X("WrapperPass", "A Pass to wrap CUDA functions with LLVM IR wrappers",
                                false, false);

// Register for clang
static void registerWrapperPass(const PassManagerBuilder &,
                         legacy::PassManagerBase &PM) {
    bool useMemorySchedule = false;
#ifdef USE_MEMORY_SCHEDULE
    useMemorySchedule = true;
#endif
    PM.add(new TaskAnalyzerPass(useMemorySchedule));
    PM.add(new WrapperPass());
}

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_OptimizerLast, registerWrapperPass);
