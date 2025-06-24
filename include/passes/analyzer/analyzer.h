#ifndef ANALYZER_H
#define ANALYZER_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/PassManager.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <stack>
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
#include <llvm/Support/raw_ostream.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/PostDominators.h>

using namespace llvm;

const std::string kernelInfoFileName = "kernel_info.txt";

struct KernelInvoke;
struct StreamGraphEdge;

struct Dim {
    CallInst *ctor;
    Value *dim3;
    SmallVector<Value *, 3> args;

    Dim() {}
    Dim(CallInst *instr): ctor(instr) {}

    void setDim3(Value *val) {
        dim3 = val;
    }
    SmallVector<Value *, 3> getArgs() { return args; }
    CallInst *getCtor() { return ctor; }
    void retriveCtor();
    void retriveDim() {
        args.clear();
        args.push_back(ctor->getArgOperand(1));
        args.push_back(ctor->getArgOperand(2));
        args.push_back(ctor->getArgOperand(3));
    }
    Value *getTotalSize();
    void moveCtorBefore(Instruction *insertPoint);
};

struct MemoryObj {
    CallInst *malloc;
    CallInst *free;
    Value *target;
    Value *size;
    SmallVector<CallInst *> memcpyList;
    SmallVector<Instruction *> mallocRelatedInstrList;
    SmallVector<Instruction *> freeRelatedInstrList;
    SmallVector<KernelInvoke *> kernelInvokes;
    std::unordered_map<CallInst *, SmallVector<Instruction *> > memcpyRelatedInstrMap;

    MemoryObj(CallInst *instr) {
        setMalloc(instr);
    }

    void retriveTarget();
    void retriveSize();
    void retriveFree();
    void retriveMemcpy();
    void clear();
    CallInst *getMalloc() { return malloc; }
    CallInst *getFree() { return free; }
    Value *getTarget() { return target; }
    Value *getSize() { return size; }
    void setMalloc(CallInst *instr) {
        clear();
        malloc = instr;
        retriveTarget();
        retriveSize();
        retriveFree();
        retriveMemcpy();
    }
    void setFree(CallInst *instr) {
        free = instr;
    }
    SmallVector<CallInst *> getMemcpyList() { return memcpyList; }
    SmallVector<Instruction *> getMallocRelatedInstrList() { return mallocRelatedInstrList; }
    SmallVector<Instruction *> getFreeRelatedInstrList() { return freeRelatedInstrList; }
    SmallVector<Instruction *> getMemcpyRelatedInstrList(CallInst *call) {
        if (memcpyRelatedInstrMap.count(call) == 0) {
            assert(false && "call in memcpyRelatedInstrMap not found");
            return SmallVector<Instruction *>();
        }
        return memcpyRelatedInstrMap[call];
    }
    SmallVector<KernelInvoke *> getKernelInvokeList() { return kernelInvokes; }
    void appendKernelInvoke(KernelInvoke *invoke) { kernelInvokes.push_back(invoke); }
};

struct KernelInvoke {
    Dim gridDim;
    Dim blockDim;
    CallInst *invoke;
    CallInst *cudaPush;
    SmallVector<MemoryObj *> memoryObjs;
    GlobalVariable *stream;
    uint32_t numRegister;
    uint32_t numSharedMem;

    KernelInvoke() {}
    KernelInvoke(CallInst *invoke, CallInst *cudaPush, GlobalVariable *stream=nullptr): 
        invoke(invoke), cudaPush(cudaPush), stream(stream) {}

    CallInst *getInvoke() { return invoke; }
    CallInst *getCudaPush() { return cudaPush; }
    void appendMemoryObj(MemoryObj *obj) { memoryObjs.push_back(obj); }
};

enum NodeKind {
    KERNEL_INVOKE,
    CUDA_MEMNODE,
    CUDA_STREAM_WAIT_EVENT
};

enum MemNodeKind {
    CUDA_MEMCPY,
    CUDA_MEMSET,    // Not supported
    CUDA_MALLOC,
    CUDA_FREE,
    CUDA_MALLOC_MEMCPY,
    CUDA_MEMCPY_FREE
};

struct StreamGraphNode {
    CallInst *call;
    NodeKind kind;
    GlobalVariable *stream;
    // SmallVector<StreamGraphEdge *> inEdges;
    SmallVector<StreamGraphEdge *> outEdges;

    StreamGraphNode(): call(nullptr) {}
    StreamGraphNode(CallInst *call): call(call) {}
    StreamGraphNode(CallInst *call, NodeKind kind, GlobalVariable *stream): 
        call(call), kind(kind), stream(stream) {}

    // void addInEdge(StreamGraphEdge *edge) { inEdges.push_back(edge); }
    void addOutEdge(StreamGraphEdge *edge) { outEdges.push_back(edge); }
    StreamGraphEdge *getOutEdge(StreamGraphNode *node);
    StreamGraphEdge *getImplicitOutEdge();
    void eraseOutEdge(StreamGraphEdge *edge);
    void setStream(GlobalVariable *stream);
    void printCall();
};

// StreamGraphMemNode contain cudaMemcpyAsync (call) and cudaMalloc/cudaFree of memory object
// If call is nullptr, it means that this node contains cudaMalloc/cudaFree only (depends on memcpyKind)
struct StreamGraphMemNode : StreamGraphNode {
    MemoryObj *memoryObj;
    enum cudaMemcpyKind memcpyKind;
    enum MemNodeKind memNodeKind;

    StreamGraphMemNode(CallInst *memcpy, GlobalVariable *stream): 
        StreamGraphNode(memcpy, CUDA_MEMNODE, stream), memoryObj(nullptr), memNodeKind(CUDA_MEMCPY) {}
    static bool classof(const StreamGraphNode *node) { return node->kind == CUDA_MEMNODE; }

    void retriveMemoryObj(SmallVector<MemoryObj *> memObjList);
    void retriveMemcpyKind();
    void setMemoryObj(MemoryObj *obj) { memoryObj = obj; }
    void setMemcpyKind(enum cudaMemcpyKind kind) { memcpyKind = kind; }
    void setMemNodeKind(enum MemNodeKind kind) { memNodeKind = kind; }
};

struct StreamGraphEdge {
    AllocaInst *event;
    StreamGraphNode *src;
    StreamGraphNode *dst;
    CallInst *createCall;
    CallInst *recordCall;
    CallInst *waitCall;
    SmallVector<Instruction *> recordRelatedInstrList;
    SmallVector<Instruction *> waitRelatedInstrList;

    StreamGraphEdge(): event(nullptr) {}
    StreamGraphEdge(AllocaInst *event): event(event) {
        retriveCreateInstr();
    }

    void setEdge(StreamGraphNode *srcNode, StreamGraphNode *dstNode) {
        setSrc(srcNode);
        setDst(dstNode);
    }
    void setSrc(StreamGraphNode *srcNode);
    void setDst(StreamGraphNode *dstNode);
    void retriveCreateInstr();
    void retriveRecordInstr();
    void retriveWaitInstr();
    void deleteEdge();
    void printDst();
};

struct MemoryGraph;
struct AnalysisResult;

struct StreamGraph {
    StreamGraph() {}

    void setResult(AnalysisResult *result);
    void addNode(StreamGraphNode *node);
    void addEdge(StreamGraphEdge *edge);
    void addStream(GlobalVariable *stream);
    int getStreamID(GlobalVariable *stream);
    GlobalVariable *getStream(int id);
    int getNumStream();
    StreamGraphNode *getNode(CallInst *call);
    SmallVector<StreamGraphNode *> getStreamVector(GlobalVariable *stream);
    SmallVector<StreamGraphNode *> getStreamVector(int id);
    StreamGraphNode *getStreamBackNode(GlobalVariable *stream);
    StreamGraphEdge *getEdge(AllocaInst *event);
    SmallVector<StreamGraphEdge *> getInEdges(StreamGraphNode *node);
    SmallVector<StreamGraphNode *> getSequenceGraph();
    GlobalVariable *retriveStream(Value *val);
    AllocaInst *retriveEvent(Value *val);
    void deleteNode(StreamGraphNode *node);
    void deleteEdgeFromGraph(StreamGraphEdge *edge);
    void deleteEdgeIfRedundant(StreamGraphEdge *edge);
    void deleteAllRedundantEdges();
    bool isDependent(StreamGraphNode *nodeA, StreamGraphNode *nodeB);
    void moveNodeBefore(StreamGraphNode *nodeA, StreamGraphNode *nodeB);
    void moveNodeAfter(StreamGraphNode *nodeA, StreamGraphNode *nodeB);
    void printStream();
    void printStreamGraph();
    void printStreamEdge();
    MemoryGraph toMemoryGraph();
    int findLastMemoryNodeID(int idx, const std::vector<std::vector<bool> > &memoryNodeVec);
    int findNextMemoryNodeID(int idx, const std::vector<std::vector<bool> > &memoryNodeVec);
    void generateStreamGraphDotFile(std::string filename="stream_graph.dot");
    int getSequenceGraphIndex(StreamGraphNode *node);

private:
    std::unordered_map<GlobalVariable *, int> streamIDMap;
    std::unordered_map<int, GlobalVariable *> IDStreamMap;
    std::unordered_map<CallInst *, StreamGraphNode *> callNodeMap;
    std::unordered_map<AllocaInst *, StreamGraphEdge *> streamEdgeMap;
    std::unordered_map<MemoryObj *, SmallVector<StreamGraphNode *>> memoryDepNodeMap;
    SmallVector<SmallVector<StreamGraphNode *> > graph;
    SmallVector<StreamGraphNode *> sequenceGraph;
    AnalysisResult *associatedResult;
};

struct MemoryGraph {
public:
    MemoryGraph(std::vector<int> streamIDs, std::vector<Value *> weights, std::vector<int> src, 
        std::vector<int> dst, int numNode, int numEdge, int numStream): streamIDs(streamIDs), weights(weights) {
        initialize(src, dst, numNode, numEdge, numStream);
    }

    void initialize(std::vector<int> src,  std::vector<int> dst, int numNode, int numEdge, int numStream);
    uint64_t getMaxMemory();
    SmallVector<Value *, 3> toLLVMArray(Instruction *insertPoint);
    bool checkDependency(const std::vector<int> &nodeList);
    void generateMemoryGraphDotFile(std::string filename="memory_graph.dot");
private:
    int numNode;
    int numEdge;
    int numStream;
    std::vector<int> streamIDs;
    std::vector<Value *> weights;
    uint64_t originalMaxMemory;
    std::vector<std::vector<bool> > adjMatrix;
    std::vector<std::vector<bool> > reachMatrix;
    std::vector<std::vector<Value *> > weightLists;
};

struct AnalysisResult {
    uint32_t numBlocks;
    uint32_t numThreadsPerBlock;
    uint64_t memoryAlloc;
    SmallVector<CallInst *> kernelInvokes;
    SmallVector<CallInst *> cudaCalls;
    SmallVector<CallInst *> cudaStreamCreateCalls;
    SmallVector<CallInst *> cudaEventCreateCalls;
    std::unordered_map<CallInst *, KernelInvoke *> kernelInvokeMap;
    std::unordered_map<CallInst *, SmallVector<CallInst *, 1>> mallocMap;
    std::unordered_map<CallInst *, SmallVector<Value *, 4>> threadSizeMap;
    std::unordered_map<Value *, Value *> mallocSizeMap;

    StreamGraph streamGraph;
    SmallVector<MemoryObj *> memObjList;
    bool isLazy;

    AnalysisResult() {}
    AnalysisResult(uint32_t grids, uint32_t blocks, uint64_t memory):
        numBlocks(grids), numThreadsPerBlock(blocks), memoryAlloc(memory) {}
    ~AnalysisResult();

    KernelInvoke *getKernelInvoke(CallInst *call);
};

struct TaskAnalyzerPass: public ModulePass {
    static char ID;

    TaskAnalyzerPass(bool useMemorySchedule);
    ~TaskAnalyzerPass(); 

    virtual bool runOnModule(Module &M) override;
    virtual void getAnalysisUsage(AnalysisUsage &AU) const override;
    void initialization(Module &M);

    AnalysisResult getResult();
    void collectKernelInvokes(Module &M);
    void collectCudaCalls(Module &M);
    bool isArgMallocRelated(Value *arg);
    CallInst *searchMallocByArg(Value *arg);
    bool analyzeMemoryRequirement(Module &M);
    bool analyzeThreadRequirement(Module &M);
    bool analyzeIntraSMRequirement(Module &M);
    bool postponeMemoryOperations(MemoryObj *memObj, 
        SmallVector<StreamGraphNode *> invokeNodeList, Module &M);
    bool preponeMemoryOperations(MemoryObj *memObj, 
        SmallVector<StreamGraphNode *> invokeNodeList, Module &M);
    bool scheduleMemoryOperations(Module &M);
    bool insertMemoryRequirement(Module &M);
    bool insertLazySchedule(Module &M);
    void insertCudaLaunchKernelPrepare(Module &M);
    void insertFakeAddrLookup(Module &M);
    void insertCudaInitialize(Module &M);
    void insertCudaFinalize(Module &M);

    // create CUDA related function wrappers
    void createCudaPredictPeakMemory(Module &M);
    void createCudaTaskSchedule(Module &M);
    void createCudaTaskScheduleLazy(Module &M);
    void createCudaLaunchKernelPrepare(Module &M);
    void createFakeAddrLookup(Module &M);
    void createCudaMallocFunc(Module &M);
    void createCudaMallocAsyncFunc(Module &M);
    void createCudaFreeAsyncFunc(Module &M);
    void createCudaEventCreateFunc(Module &M);
    void createCudaEventRecordFunc(Module &M);
    void createCudaStreamWaitEventFunc(Module &M);
    void createCudaMemPoolCreateFunc(Module &M);
    void createCudaMemPoolSetAttribute(Module &M);
    void createCudaMemPoolMalloc(Module &M);
    void createCudaStreamSetMemPool(Module &M);

    // Util
    bool isDim3Struct(Type *Ty);
    Instruction *loadStreamBefore(GlobalVariable *stream, Instruction *insertPoint, LLVMContext &context);
    AllocaInst *insertSyncBetween(Instruction *srcInsertPt, 
        GlobalVariable *srcStream, Instruction *dstInsertPt, GlobalVariable *dstStream, LLVMContext &context);
    bool removeDeadCode(Module &M);

private:
    AnalysisResult result;
    FunctionCallee cudaPredictPeakMemory;
    FunctionCallee cudaTaskSchedule;
    FunctionCallee cudaTaskScheduleLazy;
    FunctionCallee cudaLaunchKernelPrepare;
    FunctionCallee fakeAddrLookup;
    FunctionCallee cudaMallocFunc;
    FunctionCallee cudaMallocAsyncFunc;
    FunctionCallee cudaFreeAsyncFunc;
    FunctionCallee cudaEventCreateFunc;
    FunctionCallee cudaEventRecordFunc;
    FunctionCallee cudaStreamWaitEventFunc;
    FunctionCallee cudaMemPoolCreateFunc;
    FunctionCallee cudaMemPoolSetAttributeFunc;
    FunctionCallee cudaMemPoolMallocFunc;
    FunctionCallee cudaStreamSetMemPoolFunc;

    DominatorTree DT;
    PostDominatorTree PDT;

    bool useMemorySchedule;
};

std::string getDemangledName(std::string mangledName);

#endif
