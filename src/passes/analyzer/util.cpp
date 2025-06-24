#include "passes/analyzer/analyzer.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <queue>
#include <set>
#include <map>
#include <string>

/***********************************************************************************
 *                                       Util
 ***********************************************************************************
 */

std::string getDemangledName(std::string mangledName) {
  ItaniumPartialDemangler IPD;
  if (IPD.partialDemangle(mangledName.c_str())) return mangledName;

  if (IPD.isFunction())
    return IPD.getFunctionBaseName(nullptr, nullptr);
  else
    return IPD.finishDemangle(nullptr, nullptr);
}

void Dim::retriveCtor() {
    assert(dim3 && "dim3 should not be null when retriving ctor");
    for (auto user: dim3->users()) {
        if (auto CI = dyn_cast<CallInst>(user)) {
            auto name = getDemangledName(CI->getCalledFunction()->getName().str());
            if (name == "dim3") {
                ctor = CI;
                return;
            }
        }
    }
    assert(false && "Ctor is not found");
}

Value *Dim::getTotalSize() {
    assert(ctor && "ctor should not be null when retriving total size");
    retriveDim();
    assert(args.size() > 0 && "args should not be empty when retriving total size");
    Value *totalSize = args[0];
    Instruction *insertPoint = ctor->getNextNode();
    for (int i = 1; i < args.size(); i++) {
        totalSize = BinaryOperator::CreateMul(totalSize, args[i], "totalSize", insertPoint);
    }
    return totalSize;
}

void Dim::moveCtorBefore(Instruction *insertPoint) {
    assert(ctor && "ctor should not be null when moving before insert point");
    if (insertPoint == nullptr) {
        return;
    }
    std::queue<Value *> workQueue;
    SmallPtrSet<Value *, 16> visited;
    std::stack<Instruction *> instrStack;
    workQueue.push(ctor);
    visited.insert(ctor);

    while (!workQueue.empty()) {
        Value *current = workQueue.front();
        workQueue.pop();
        if (auto *I = dyn_cast<Instruction>(current)) {
            instrStack.push(I);
        }

        // Traverse the users of the current value
        for (auto *user : current->users()) {
            if (visited.insert(user).second) {
                workQueue.push(user);
            }
        }
    }

    while (!instrStack.empty()) {
        Instruction *cur = instrStack.top();
        instrStack.pop();
        cur->moveBefore(insertPoint);
    }
    // dbgs() << "Move dim3 ctor" << *ctor << " before insert point: " << *insertPoint << "\n";
}

void MemoryObj::retriveSize() {
    Function *Callee = malloc->getCalledFunction();
    auto name = getDemangledName(Callee->getName().str());

    if (name != "cudaMalloc" && name != "cudaMallocAsync") {
        assert(false && "malloc instruction is not found");
    }

    // Direct cudaMalloc call
    if (malloc->arg_size() == 2 || name == "cudaMallocAsync") {
        size = malloc->getArgOperand(1);
        return;
    }
    // cudaMalloc Wrapper function
    else {
        for (inst_iterator I = inst_begin(Callee), E = inst_end(Callee); I != E;
            ++I) {
            if (auto CI = dyn_cast<CallInst>(&*I)) {
                auto n = getDemangledName(CI->getCalledFunction()->getName().str());
                if (n == "cudaMalloc" || n == "cudaMallocAsync") {
                    size = CI->getArgOperand(1);
                    return;
                }
            }
        }
        assert(false && "malloc size is not found for cudaMalloc Wrapper");
    }
}

void MemoryObj::retriveTarget() {
    assert(malloc && "Malloc should not be empty when retriving target");
    std::stack<Instruction *> instrStack;
    Value *ans = malloc->getArgOperand(0);
    instrStack.push(malloc);
    if (isa<LoadInst>(ans)) {
        instrStack.push(dyn_cast<LoadInst>(ans));
        ans = dyn_cast<LoadInst>(ans)->getOperand(0);
    }
    if (isa<BitCastInst>(ans)) {
        instrStack.push(dyn_cast<BitCastInst>(ans));
        ans = dyn_cast<BitCastInst>(ans)->getOperand(0);
    }
    target = ans;

    // Collect cudaMalloc related instructions
    while(!instrStack.empty()) {
        mallocRelatedInstrList.push_back(instrStack.top());
        instrStack.pop();
    }
}

void MemoryObj::retriveFree() {
    assert(malloc && target && "Malloc should not be empty when retriving free");
    Value *ans = target;
    for (auto user : target->users()) {
        if (auto bitcast = dyn_cast<BitCastInst>(user)) {
            for (auto bitcast_user : bitcast->users()) {
                if (auto load = dyn_cast<LoadInst>(bitcast_user)) {
                    for (auto load_user : load->users()) {
                        if (auto call = dyn_cast<CallInst>(load_user)) {
                            auto name = getDemangledName(call->getCalledFunction()->getName().str());
                            if (name == "cudaFree" || name == "cudaFreeAsync") {
                                free = call;
                                freeRelatedInstrList.push_back(bitcast);
                                freeRelatedInstrList.push_back(load);
                                freeRelatedInstrList.push_back(call);
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
    assert(false && "Free instruction is not found for memory object");
}

void MemoryObj::retriveMemcpy() {
    assert(malloc && target && "Malloc and target should not be empty when retriving cudaMemcpy");
    Value *ans = target;
    for (auto user : target->users()) {
        if (auto bitcast = dyn_cast<BitCastInst>(user)) {
            for (auto bitcast_user : bitcast->users()) {
                if (auto load = dyn_cast<LoadInst>(bitcast_user)) {
                    for (auto load_user : load->users()) {
                        if (auto call = dyn_cast<CallInst>(load_user)) {
                            auto name = getDemangledName(call->getCalledFunction()->getName().str());
                            if (name == "cudaMemcpy" || name == "cudaMemcpyAsync") {
                                memcpyList.push_back(call);
                                memcpyRelatedInstrMap[call].push_back(bitcast);
                                memcpyRelatedInstrMap[call].push_back(load);
                                memcpyRelatedInstrMap[call].push_back(call);
                            }
                        }
                    }
                }
            }
        }
    }
    if (memcpyList.size() == 0) {
        // dbgs() << "Warning: cudaMemcpy instruction is not found for memory object\n";
    }
}

void MemoryObj::clear() {
    size = nullptr;
    target = nullptr;
    free = nullptr;
    memcpyList.clear();
    mallocRelatedInstrList.clear();
    freeRelatedInstrList.clear();
    memcpyRelatedInstrMap.clear();
}

StreamGraphEdge *StreamGraphNode::getOutEdge(StreamGraphNode *dstNode) {
    StreamGraphEdge *result = nullptr;
    for (auto edge : outEdges) {
        if (edge->dst == dstNode) {
            result = edge;
            break;
        }
    }
    return result;
}

StreamGraphEdge *StreamGraphNode::getImplicitOutEdge() {
    StreamGraphEdge *implicitEdge = nullptr;
    for (auto edge : outEdges) {
        if (edge->event == nullptr) {
            implicitEdge = edge;
            break;
        }
    }
    return implicitEdge;
}

void StreamGraphNode::eraseOutEdge(StreamGraphEdge *edge) {
    if (edge == nullptr) {
        return;
    }
    bool flag = true;
    while (flag) {
        flag = false;
        for (int i = 0; i < outEdges.size(); i ++) {
            if (edge == outEdges[i]) {
                // dbgs() << "Erase edge: ";
                if (edge->src->call) {
                    // dbgs() << *(edge->src->call);
                }
                else {
                    // dbgs() << "streamGraphNode without call";
                }
                // dbgs() << " -> ";
                if (edge->dst->call) {
                    // dbgs() << *(edge->dst->call);
                }
                else {
                    // dbgs() << "streamGraphNode without call";
                }
                // dbgs() << "\n";
                outEdges.erase(outEdges.begin() + i);
                flag = true;
                break;
            }
        }
    }
}

void StreamGraphNode::setStream(GlobalVariable *stream) {
    if (stream == this->stream) {
        return;
    }

    // All outEdges should be update if stream is changed
    for (auto edge : outEdges) {
        // Implicit edge
        if (edge->event == nullptr) {
            continue;
        }
        
        SmallVector<Instruction *> recordRelatedInstrList;
        CallInst *recordCall = nullptr;
        LoadInst *loadStreamInstr = nullptr;
        // Find record and load stream instruction
        for (auto instr : edge->recordRelatedInstrList) {
            if (auto call = dyn_cast<CallInst>(instr)) {
                recordCall = call;
            }
        }
        assert(recordCall && "Record call is not found");
        for (auto instr : edge->recordRelatedInstrList) {
            if (auto load = dyn_cast<LoadInst>(instr)) {
                loadStreamInstr = load;
            }
        }
        assert(loadStreamInstr && "Load stream instruction is not found");

        // Replace stream in record call
        recordCall->setArgOperand(1, stream);
    }


    this->stream = stream;
}

void StreamGraphNode::printCall() {
    if (call) {
        // dbgs() << *call;
    }
    else {
        // dbgs() << "streamGraphNode without call";
    }
}

void StreamGraphMemNode::retriveMemoryObj(SmallVector<MemoryObj *> memObjList) {
    assert(call && "Call instruction should not be empty when retriving memory object");
    for (auto memObj : memObjList) {
        for (auto memcpy : memObj->getMemcpyList()) {
            if (call == memcpy) {
                memoryObj = memObj;
                return;
            }
        }
    }
    assert(false && "Memory object is not found");
}

void StreamGraphMemNode::retriveMemcpyKind() {
    assert(call && "Call instruction should not be empty when retriving memcpy kind");
    auto kindArg = dyn_cast<ConstantInt>(call->getArgOperand(3));
    assert(kindArg && "Memcpy kind argument should be a constant integer");
    memcpyKind = (enum cudaMemcpyKind)kindArg->getZExtValue();
}

void StreamGraphEdge::setSrc(StreamGraphNode *srcNode) {
    assert(srcNode != nullptr && "'srcNode' in setSrc should not be nullptr");
    src = srcNode;
    recordRelatedInstrList.clear();
    retriveRecordInstr();
    bool flag = false;
    for (auto edge : src->outEdges) {
        if (edge == this) {
            flag = true;
            break;
        }
    }
    if (!flag) {
        src->addOutEdge(this);
    }
}

void StreamGraphEdge::setDst(StreamGraphNode *dstNode) {
    assert(dstNode != nullptr && "'dstNode' in setSrc should not be nullptr");
    dst = dstNode;
    waitRelatedInstrList.clear();
    retriveWaitInstr();
    if (src) {
        bool flag = false;
        for (auto edge : src->outEdges) {
            if (edge == this) {
                flag = true;
                break;
            }
        }
        if (!flag) {
            src->addOutEdge(this);
        }
    }
}

void StreamGraphEdge::retriveCreateInstr() {
    // Edge is not associated with any event (nodes in the same stream)
    if (event == nullptr) {
        assert(false && "Edge is not associated with any event");
    }
    // // dbgs() << "Enter retriveCreateInstr\n";
    for (auto user : event->users()) {
        if (auto call = dyn_cast<CallInst>(user)) {
            auto name = getDemangledName(call->getCalledFunction()->getName().str());
            if (name == "cudaEventCreate") {
                createCall = call;
                // dbgs() << "edge event retriveCreateInstr: " << *call << "\n";
                return;
            }
        }
    }
    assert(false && "retriveCreateInstr instruction is not found");
}

void StreamGraphEdge::retriveRecordInstr() {
    // Edge is not associated with any event (nodes in the same stream)
    if (event == nullptr) {
        return;
    }
    for (auto user : event->users()) {
        if (auto load = dyn_cast<LoadInst>(user)) {
            for (auto load_user : load->users()) {
                if (auto call = dyn_cast<CallInst>(load_user)) {
                    auto name = getDemangledName(call->getCalledFunction()->getName().str());
                    if (name == "cudaEventRecord") {
                        // dbgs() << "edge event retriveRecordInstr: " << *call << "\n";
                        recordRelatedInstrList.push_back(load);
                        assert(isa<LoadInst>(call->getArgOperand(1)));
                        recordRelatedInstrList.push_back(dyn_cast<LoadInst>(call->getArgOperand(1)));
                        recordRelatedInstrList.push_back(call);
                        recordCall = call;
                        return;
                    }
                }
            }
        }
    }
    assert(false && "cudaEventRecord instruction is not found");
}

void StreamGraphEdge::retriveWaitInstr() {
    // Edge is not associated with any event (nodes in the same stream)
    if (event == nullptr) {
        return;
    }
    for (auto user : event->users()) {
        if (auto load = dyn_cast<LoadInst>(user)) {
            for (auto load_user : load->users()) {
                if (auto call = dyn_cast<CallInst>(load_user)) {
                    auto name = getDemangledName(call->getCalledFunction()->getName().str());
                    if (name == "cudaStreamWaitEvent") {
                        assert(isa<LoadInst>(call->getArgOperand(0)));
                        waitRelatedInstrList.push_back(dyn_cast<LoadInst>(call->getArgOperand(0)));
                        waitRelatedInstrList.push_back(load);
                        waitRelatedInstrList.push_back(call);
                        waitCall = call;
                        return;
                    }
                }
            }
        }
    }
    assert(false && "cudaStreamWaitEvent instruction is not found");
}

// Delete edge and update the inEdges and outEdges for its source and destination node
void StreamGraphEdge::deleteEdge() {
    if (src != nullptr) {
        for (int i = 0; i < src->outEdges.size(); i ++) {
            if (src->outEdges[i] == this) {
                src->outEdges.erase(src->outEdges.begin() + i);
                i = -1;
            }
        }
    }
}

void StreamGraphEdge::printDst() {
    // dbgs() << "    -> ";
    if (event == nullptr) {
        // dbgs() << "[Implicit Edge] ";
    }
    if (dst) {
        dst->printCall();
    }
    else {
        // dbgs() << "streamGraphEdge without dst";
    }
    // dbgs() << "\n";
}

void StreamGraph::setResult(AnalysisResult *result) {
    associatedResult = result;
}

void StreamGraph::addNode(StreamGraphNode *node) {
    assert(node->stream != nullptr && "Add node without setting stream");
    StreamGraphNode *backNode = getStreamBackNode(node->stream);
    int streamID = getStreamID(node->stream);
    // One or more event nodes (cudaStreamWaitEvent) are inserted to add edge
    if (node->kind != CUDA_STREAM_WAIT_EVENT && backNode && backNode->kind == CUDA_STREAM_WAIT_EVENT) {
        std::stack<StreamGraphNode *> nodeStack;
        while (!graph[streamID].empty() && graph[streamID].back()->kind == CUDA_STREAM_WAIT_EVENT) {
            nodeStack.push(graph[streamID].back());
            graph[streamID].pop_back();
            if (auto predNode = getStreamBackNode(node->stream)) {
                StreamGraphEdge *implicitEdge = predNode->getImplicitOutEdge();
                deleteEdgeFromGraph(implicitEdge);
            }
        }
        while (!nodeStack.empty()) {
            StreamGraphNode *eventNode = nodeStack.top();
            CallInst *eventWaitCall = eventNode->call;
            AllocaInst *event = retriveEvent(eventWaitCall->getArgOperand(1));
            StreamGraphEdge *edge = getEdge(event);
            edge->setDst(node);
            nodeStack.pop();
        }
    }
    // Add edge for nodes in the same stream
    if (auto predNode = getStreamBackNode(node->stream)) {
        StreamGraphEdge *edge = new StreamGraphEdge();
        edge->setEdge(predNode, node);
    }
    graph[streamID].push_back(node);
    callNodeMap[node->call] = node;
    if (node->kind != CUDA_STREAM_WAIT_EVENT) {
        sequenceGraph.push_back(node);
    }
}

void StreamGraph::addEdge(StreamGraphEdge *edge) {
    if (streamEdgeMap.find(edge->event) != streamEdgeMap.end()) {
        // dbgs() << "[Error]: Edge already exists\n";
        if (edge->event) {
            // dbgs() << "Event: " << *(edge->event) << "\n";
            if (edge->src->call) {
                // dbgs() << "Src: " << *(edge->src->call) << "\n";
            }
            else {
                // dbgs() << "Src: streamGraphNode without call\n";
            }
            if (edge->dst->call) {
                // dbgs() << "Dst: " << *(edge->dst->call) << "\n";
            }
            else {
                // dbgs() << "Dst: streamGraphNode without call\n";
            }
        }
        else {
            // dbgs() << "Event is nullptr\n";
        }
    }
    assert(streamEdgeMap.find(edge->event) == streamEdgeMap.end());
    streamEdgeMap[edge->event] = edge;
}

void StreamGraph::addStream(GlobalVariable *stream) {
    assert(stream->getName().contains("mzw_s"));
    streamIDMap[stream] = std::stoi(stream->getName().str().substr(5));
    IDStreamMap[std::stoi(stream->getName().str().substr(5))] = stream;
    graph.push_back(SmallVector<StreamGraphNode *>());
}

int StreamGraph::getStreamID(GlobalVariable *stream) {
    return streamIDMap[stream];
}

GlobalVariable *StreamGraph::getStream(int id) {
    if (IDStreamMap.find(id) == IDStreamMap.end()) {
        return nullptr;
    }
    return IDStreamMap[id];
}

int StreamGraph::getNumStream() {
    return streamIDMap.size();
}

StreamGraphNode *StreamGraph::getNode(CallInst *call) {
    if (callNodeMap.find(call) == callNodeMap.end()) {
        return nullptr;
    }
    return callNodeMap[call];
}

SmallVector<StreamGraphNode *> StreamGraph::getStreamVector(GlobalVariable *stream) {
    return graph[streamIDMap[stream]];
}

SmallVector<StreamGraphNode *> StreamGraph::getStreamVector(int id) {
    return graph[streamIDMap[getStream(id)]];
}

StreamGraphNode *StreamGraph::getStreamBackNode(GlobalVariable *stream) {
    if (graph[streamIDMap[stream]].empty()) {
        return nullptr;
    }
    return graph[streamIDMap[stream]].back();
}

StreamGraphEdge *StreamGraph::getEdge(AllocaInst *event) {
    assert(streamEdgeMap.find(event) != streamEdgeMap.end() && "Edge associated with event is not found");
    return streamEdgeMap[event];
}

SmallVector<StreamGraphEdge *> StreamGraph::getInEdges(StreamGraphNode *node) {
    SmallVector<StreamGraphEdge *> inEdges;
    for (auto pair : streamEdgeMap) {
        if (pair.second->dst == node) {
            inEdges.push_back(pair.second);
        }
    }
    return inEdges;
}

SmallVector<StreamGraphNode *> StreamGraph::getSequenceGraph() {
    return sequenceGraph;
}

void StreamGraph::deleteNode(StreamGraphNode *node) {
    if (node == nullptr) {
        return;
    }
    int streamID = getStreamID(node->stream);
    for (int i = 0; i < graph[streamID].size(); i ++) {
        if (graph[streamID][i] == node) {
            // Erase implicit out edge for last node in the same stream
            if (i > 0) {
                StreamGraphNode *src = graph[streamID][i - 1];
                for (int j = 0; j < src->outEdges.size(); j ++) {
                    StreamGraphEdge *outEdge = src->outEdges[j];
                    if (outEdge && outEdge->event == nullptr) {
                        src->outEdges.erase(src->outEdges.begin() + j);
                        deleteEdgeFromGraph(outEdge);
                        outEdge->deleteEdge();
                        break;
                    }
                }
            }
            graph[streamID].erase(graph[streamID].begin() + i);
            break;
        }
    }
    for (int i = 0; i < sequenceGraph.size(); i ++) {
        if (sequenceGraph[i] == node) {
            sequenceGraph.erase(sequenceGraph.begin() + i);
            break;
        }
    }
    delete node;
}

void StreamGraph::deleteEdgeFromGraph(StreamGraphEdge *edge) {
    if (edge == nullptr) {
        return;
    }
    if (edge->src != nullptr) {
        edge->src->eraseOutEdge(edge);
    }
    if (streamEdgeMap.find(edge->event) != streamEdgeMap.end()) {
        streamEdgeMap.erase(edge->event);
    }
}

void StreamGraph::deleteEdgeIfRedundant(StreamGraphEdge *edge) {
    if (edge->event == nullptr) {
        return;
    }

    deleteEdgeFromGraph(edge);
            
    // Remove synchronization-related instructions if the edge is redundant
    if (isDependent(edge->src, edge->dst)) {
        deleteEdgeFromGraph(edge);
        edge->deleteEdge();
        // dbgs() << "Remove redundant edge: " << *(edge->src->call) << " -> " << *(edge->dst->call) << "\n";
        for (auto instr = edge->recordRelatedInstrList.rbegin(); instr != edge->recordRelatedInstrList.rend(); instr ++) {
            if ((*instr)->getParent() != nullptr) {
                (*instr)->eraseFromParent();
            }
        }
        edge->recordRelatedInstrList.clear();
        for (auto instr = edge->waitRelatedInstrList.rbegin(); instr != edge->waitRelatedInstrList.rend(); instr ++) {
            if ((*instr)->getParent() != nullptr) {
                (*instr)->eraseFromParent();
            }
        }
        edge->waitRelatedInstrList.clear();
        assert(edge->createCall != nullptr && "createCall should not be nullptr");
        edge->createCall->eraseFromParent();
        edge->event->eraseFromParent();
        for (auto iter = associatedResult->cudaEventCreateCalls.begin(); 
            iter != associatedResult->cudaEventCreateCalls.end(); iter ++) {
            if (*iter == edge->createCall) {
                associatedResult->cudaEventCreateCalls.erase(iter);
                break;
            }
        }
    }
    else {
        edge->setSrc(edge->src);
        addEdge(edge);
    }
}

void StreamGraph::deleteAllRedundantEdges() {
    for (auto node : sequenceGraph) {
        auto tmpEdges = node->outEdges;
        for (auto edge : tmpEdges) {
            deleteEdgeIfRedundant(edge);
        }
    }
    // dbgs() << "Finish deleting all redundant edges\n";
}

// Check if dependency 'nodeA->nodeB' exists
bool StreamGraph::isDependent(StreamGraphNode *nodeA, StreamGraphNode *nodeB) {
    std::queue<StreamGraphNode *> nodeQueue;
    std::set<StreamGraphNode *> visitedNode;
    nodeQueue.push(nodeA);
    // BFS search
    while (!nodeQueue.empty()) {
        StreamGraphNode *node = nodeQueue.front();
        nodeQueue.pop();
        if (node == nodeB) {
            return true;
        }
        else if (visitedNode.count(node) > 0) {
            continue;
        }
        visitedNode.insert(node);
        for (auto edge : node->outEdges) {
            nodeQueue.push(edge->dst);
        }
    }
    return false;
}

// Move nodeA before nodeB in stream graph and update the sequence graph
void StreamGraph::moveNodeBefore(StreamGraphNode *toMoveNode, StreamGraphNode *dstNode) {
    if (toMoveNode == nullptr || dstNode == nullptr) {
        assert(false && "toMoveNode or dstNode is nullptr");
    }
    // Already satisfied
    auto tmpEdge = toMoveNode->getImplicitOutEdge();
    if (toMoveNode == dstNode || (tmpEdge && tmpEdge->dst == dstNode)) {
        return;
    }
    bool foundSrc = false;
    bool foundDst = false;
    // Remove all out edges of toMoveNode
    SmallVector<StreamGraphEdge *> tmpOutEdges = toMoveNode->outEdges;
    for (auto edge : tmpOutEdges) {
        deleteEdgeFromGraph(edge);
        edge->deleteEdge();
        if (edge->event) {
            for (auto instr = edge->recordRelatedInstrList.rbegin(); instr != edge->recordRelatedInstrList.rend(); instr ++) {
                if ((*instr)->getParent() != nullptr) {
                    (*instr)->eraseFromParent();
                }
            }
            edge->recordRelatedInstrList.clear();
            for (auto instr = edge->waitRelatedInstrList.rbegin(); instr != edge->waitRelatedInstrList.rend(); instr ++) {
                if ((*instr)->getParent() != nullptr) {
                    (*instr)->eraseFromParent();
                }
            }
            edge->waitRelatedInstrList.clear();
            assert(edge->createCall != nullptr && "createCall should not be nullptr");
            if (edge->createCall->getParent() != nullptr) {
                edge->createCall->eraseFromParent();
                for (auto iter = associatedResult->cudaEventCreateCalls.begin(); 
                    iter != associatedResult->cudaEventCreateCalls.end(); iter ++) {
                    if (*iter == edge->createCall) {
                        associatedResult->cudaEventCreateCalls.erase(iter);
                        break;
                    }
                }
            }
            if (edge->event->getParent() != nullptr) {
                edge->event->eraseFromParent();
            }
        }
    }
    // Erase toMoveNode from graph
    int streamID = getStreamID(toMoveNode->stream);
    for (int i = 0; i < graph[streamID].size(); i ++) {
        if (graph[streamID][i] == toMoveNode) {
            // Update implicit edge in the same stream
            if (i > 0) {
                StreamGraphEdge *implicitOutEdge = graph[streamID][i - 1]->getImplicitOutEdge();
                if (implicitOutEdge) {
                    deleteEdgeFromGraph(implicitOutEdge);
                    implicitOutEdge->deleteEdge();
                }
                else {
                    assert(false && "Implicit edge must exists if not the last node");
                }
                if (i + 1 < graph[streamID].size()) {
                    StreamGraphEdge *edge = new StreamGraphEdge();
                    edge->setEdge(graph[streamID][i - 1], graph[streamID][i + 1]);
                }
            }
            graph[streamID].erase(graph[streamID].begin() + i);
            foundSrc = true;
            break;
        }
    }
    for (int i = 0; i < sequenceGraph.size(); i ++) {
        if (sequenceGraph[i] == toMoveNode) {
            sequenceGraph.erase(sequenceGraph.begin() + i);
            break;
        }
    }

    // Add toMoveNode to graph
    streamID = getStreamID(dstNode->stream);
    for (int i = 0; i < graph[streamID].size(); i ++) {
        if (graph[streamID][i] == dstNode) {
            // Update the implicit edge of lastNode in the same stream
            if (i > 0) {
                StreamGraphEdge *implicitOutEdge = graph[streamID][i - 1]->getImplicitOutEdge();
                if (implicitOutEdge) {
                    // Dst of implicit edge already exists in outEdges, transfer it to implicit edge
                    if (auto edge = graph[streamID][i - 1]->getOutEdge(toMoveNode)) {
                        for (auto instr : edge->recordRelatedInstrList) {
                            instr->eraseFromParent();
                        }
                        edge->recordRelatedInstrList.clear();
                        for (auto instr : edge->waitRelatedInstrList) {
                            instr->eraseFromParent();
                        }
                        edge->waitRelatedInstrList.clear();
                        edge->event->eraseFromParent();
                        edge->createCall->eraseFromParent();
                        for (auto iter = associatedResult->cudaEventCreateCalls.begin(); 
                            iter != associatedResult->cudaEventCreateCalls.end(); iter ++) {
                            if (*iter == edge->createCall) {
                                associatedResult->cudaEventCreateCalls.erase(iter);
                                break;
                            }
                        }
                        deleteEdgeFromGraph(edge);
                        edge->deleteEdge();
                    }
                    // Delete and create an implicit edge for last_node->toMoveNode
                    deleteEdgeFromGraph(implicitOutEdge);
                    implicitOutEdge->deleteEdge();
                    implicitOutEdge = new StreamGraphEdge();
                    implicitOutEdge->setEdge(graph[streamID][i - 1], toMoveNode);
                }
                else {
                    assert(false && "Implicit edge must exists if not the last node");
                }
            }
            graph[streamID].insert(graph[streamID].begin() + i, toMoveNode);
            foundDst = true;
            break;
        }
    }
    for (int i = 0; i < sequenceGraph.size(); i ++) {
        if (sequenceGraph[i] == dstNode) {
            sequenceGraph.insert(sequenceGraph.begin() + i, toMoveNode);
            break;
        }
    }
    if (!foundSrc || !foundDst) {
        assert(false && "toMoveNode or dstNode is not found in its stream graph");
    }

    // Update stream of toMoveNode
    toMoveNode->setStream(dstNode->stream);

    // Update implicit edge for toMoveNode
    StreamGraphEdge *implicitOutEdge = toMoveNode->getImplicitOutEdge();
    // Dst of implicit edge already exists in outEdges, transfer it to implicit edge
    if (implicitOutEdge) {
        if (auto edge = toMoveNode->getOutEdge(dstNode)) {
            for (auto instr : edge->recordRelatedInstrList) {
                instr->eraseFromParent();
            }
            edge->recordRelatedInstrList.clear();
            for (auto instr : edge->waitRelatedInstrList) {
                instr->eraseFromParent();
            }
            edge->waitRelatedInstrList.clear();
            edge->event->eraseFromParent();
            edge->createCall->eraseFromParent();
            for (auto iter = associatedResult->cudaEventCreateCalls.begin(); 
                iter != associatedResult->cudaEventCreateCalls.end(); iter ++) {
                if (*iter == edge->createCall) {
                    associatedResult->cudaEventCreateCalls.erase(iter);
                    break;
                }
            }
            deleteEdgeFromGraph(edge);
            edge->deleteEdge();
        }
        deleteEdgeFromGraph(implicitOutEdge);
        implicitOutEdge->deleteEdge();
    }
    implicitOutEdge = new StreamGraphEdge();
    implicitOutEdge->setEdge(toMoveNode, dstNode);
}

// Move nodeA after nodeB in stream graph and update the sequence graph
void StreamGraph::moveNodeAfter(StreamGraphNode *toMoveNode, StreamGraphNode *dstNode) {
    if (toMoveNode == nullptr || dstNode == nullptr) {
        assert(false && "toMoveNode or dstNode is nullptr");
    }
    // Already satisfied
    auto tmpEdge = dstNode->getImplicitOutEdge();
    if (toMoveNode == dstNode || (tmpEdge && tmpEdge->dst == toMoveNode)) {
        return;
    }
    bool foundSrc = false;
    bool foundDst = false;
    // Remove all out edges of toMoveNode
    SmallVector<StreamGraphEdge *> tmpOutEdges = toMoveNode->outEdges;
    for (auto edge : tmpOutEdges) {
        deleteEdgeFromGraph(edge);
        edge->deleteEdge();
        if (edge->event) {
            for (auto instr = edge->recordRelatedInstrList.rbegin(); instr != edge->recordRelatedInstrList.rend(); instr ++) {
                if ((*instr)->getParent() != nullptr) {
                    (*instr)->eraseFromParent();
                }
            }
            edge->recordRelatedInstrList.clear();
            for (auto instr = edge->waitRelatedInstrList.rbegin(); instr != edge->waitRelatedInstrList.rend(); instr ++) {
                if ((*instr)->getParent() != nullptr) {
                    (*instr)->eraseFromParent();
                }
            }
            edge->waitRelatedInstrList.clear();
            assert(edge->createCall != nullptr && "createCall should not be nullptr");
            if (edge->createCall->getParent() != nullptr) {
                edge->createCall->eraseFromParent();
                for (auto iter = associatedResult->cudaEventCreateCalls.begin(); 
                    iter != associatedResult->cudaEventCreateCalls.end(); iter ++) {
                    if (*iter == edge->createCall) {
                        associatedResult->cudaEventCreateCalls.erase(iter);
                        break;
                    }
                }
            }
            if (edge->event->getParent() != nullptr) {
                edge->event->eraseFromParent();
            }
        }
    }
    // Erase toMoveNode from graph
    int streamID = getStreamID(toMoveNode->stream);
    for (int i = 0; i < graph[streamID].size(); i ++) {
        if (graph[streamID][i] == toMoveNode) {
            // Update implicit edge in the same stream
            if (i > 0) {
                StreamGraphEdge *implicitOutEdge = graph[streamID][i - 1]->getImplicitOutEdge();
                if (implicitOutEdge) {
                    deleteEdgeFromGraph(implicitOutEdge);
                    implicitOutEdge->deleteEdge();
                }
                else {
                    assert(false && "Implicit edge must exists if not the last node");
                }
                if (i + 1 < graph[streamID].size()) {
                    StreamGraphEdge *edge = new StreamGraphEdge();
                    edge->setEdge(graph[streamID][i - 1], graph[streamID][i + 1]);
                }
            }
            graph[streamID].erase(graph[streamID].begin() + i);
            foundSrc = true;
            break;
        }
    }
    for (int i = 0; i < sequenceGraph.size(); i ++) {
        if (sequenceGraph[i] == toMoveNode) {
            sequenceGraph.erase(sequenceGraph.begin() + i);
            break;
        }
    }

    // Add toMoveNode to graph
    streamID = getStreamID(dstNode->stream);
    for (int i = 0; i < graph[streamID].size(); i ++) {
        if (graph[streamID][i] == dstNode) {
            // Update the implicit edge of toMoveNode
            StreamGraphEdge *implicitOutEdge = toMoveNode->getImplicitOutEdge();
            if (implicitOutEdge) {
                deleteEdgeFromGraph(implicitOutEdge);
                implicitOutEdge->deleteEdge();
            }
            if (i + 1 < graph[streamID].size()) {
                if (auto edge = toMoveNode->getOutEdge(graph[streamID][i + 1])) {
                    deleteEdgeFromGraph(edge);
                    edge->deleteEdge();
                    for (auto instr : edge->recordRelatedInstrList) {
                        instr->eraseFromParent();
                    }
                    edge->recordRelatedInstrList.clear();
                    for (auto instr : edge->waitRelatedInstrList) {
                        instr->eraseFromParent();
                    }
                    edge->waitRelatedInstrList.clear();
                    edge->event->eraseFromParent();
                    edge->createCall->eraseFromParent();
                    for (auto iter = associatedResult->cudaEventCreateCalls.begin(); 
                        iter != associatedResult->cudaEventCreateCalls.end(); iter ++) {
                        if (*iter == edge->createCall) {
                            associatedResult->cudaEventCreateCalls.erase(iter);
                            break;
                        }
                    }
                }
                // else {
                implicitOutEdge = new StreamGraphEdge();
                implicitOutEdge->setEdge(toMoveNode, graph[streamID][i + 1]);
                // }
            }
            graph[streamID].insert(graph[streamID].begin() + (i + 1), toMoveNode);
            foundDst = true;
            break;
        }
    }
    for (int i = 0; i < sequenceGraph.size(); i ++) {
        if (sequenceGraph[i] == dstNode) {
            sequenceGraph.insert(sequenceGraph.begin() + i, toMoveNode);
            break;
        }
    }

    // Update stream of toMoveNode
    // dbgs() << "toMoveNode stream: " << *(toMoveNode->stream) << "\n";
    toMoveNode->setStream(dstNode->stream);
    // dbgs() << "toMoveNode stream: " << *(toMoveNode->stream) << "\n";

    // Update implicit edge for dstNode
    StreamGraphEdge *implicitOutEdge = dstNode->getImplicitOutEdge();
    if (implicitOutEdge) {
        deleteEdgeFromGraph(implicitOutEdge);
        implicitOutEdge->deleteEdge();
    }
    if (auto edge = dstNode->getOutEdge(toMoveNode)) {
        deleteEdgeFromGraph(edge);
        edge->deleteEdge();
        for (auto instr : edge->recordRelatedInstrList) {
            instr->eraseFromParent();
        }
        edge->recordRelatedInstrList.clear();
        for (auto instr : edge->waitRelatedInstrList) {
            instr->eraseFromParent();
        }
        edge->waitRelatedInstrList.clear();
        edge->event->eraseFromParent();
        edge->createCall->eraseFromParent();
        for (auto iter = associatedResult->cudaEventCreateCalls.begin(); 
            iter != associatedResult->cudaEventCreateCalls.end(); iter ++) {
            if (*iter == edge->createCall) {
                associatedResult->cudaEventCreateCalls.erase(iter);
                break;
            }
        }
    }
    implicitOutEdge = new StreamGraphEdge();
    implicitOutEdge->setEdge(dstNode, toMoveNode);
}

Instruction *TaskAnalyzerPass::loadStreamBefore(GlobalVariable *stream, Instruction *insertPoint, LLVMContext &context) {
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);

    IRBuilder<> builder(context);
    builder.SetInsertPoint(insertPoint);
    auto load = builder.CreateLoad(streamStructPtTy, stream);
    // dbgs() << "loadStreamBefore: " << *load << "with stream " << *stream << "\n";
    return load;
}

AllocaInst *TaskAnalyzerPass::insertSyncBetween(Instruction *srcInsertPt, 
    GlobalVariable *srcStream, Instruction *dstInsertPt, GlobalVariable *dstStream, LLVMContext &context) {
    assert(srcInsertPt && dstInsertPt && "srcInsertPt or dstInsertPt is nullptr");
    assert(srcStream && dstStream && "srcStream or dstStream is nullptr");
    if (srcStream == dstStream) {
        assert(false && "srcStream and dstStream should not be the same");
    }

    // Create cudaEvent_t
    StructType *eventStructTy = StructType::getTypeByName(context, "struct.CUevent_st");
    // struct.CUevent_st is not defined in the program
    if (eventStructTy == nullptr) {
        eventStructTy = StructType::create(context, "struct.CUevent_st");
    }
    Type *eventStructPtTy = PointerType::getUnqual(eventStructTy);
    auto eventCreateInsertPoint = srcInsertPt->getFunction()->getEntryBlock().getTerminator();
    IRBuilder<> builder(context);
    builder.SetInsertPoint(eventCreateInsertPoint);
    auto dummyAlloc = builder.CreateAlloca(eventStructPtTy, nullptr, "dummyEventAlloc");
    auto eventAlloc = builder.CreateAlloca(eventStructPtTy, nullptr, "newEventAlloc");
    auto eventCreate = builder.CreateCall(cudaEventCreateFunc, {eventAlloc}, "newEventCreate");
    // dbgs() << "eventAlloc: " << *eventAlloc << "\n";
    // dbgs() << "eventCreate: " << *eventCreate << "\n";

    // Insert cudaEventRecord after srcInsertPt
    StructType *streamStructTy = StructType::getTypeByName(context, "struct.CUstream_st");
    // struct.CUstream_st is not defined in the program
    if (streamStructTy == nullptr) {
        streamStructTy = StructType::create(context, "struct.CUstream_st");
    }
    Type *streamStructPtTy = PointerType::getUnqual(streamStructTy);
    builder.SetInsertPoint(srcInsertPt->getNextNonDebugInstruction());
    auto loadEvent = builder.CreateLoad(eventStructPtTy, eventAlloc);
    auto loadStream = builder.CreateLoad(streamStructPtTy, srcStream);
    auto eventRecord = builder.CreateCall(cudaEventRecordFunc, {loadEvent, loadStream}, "newEventRecord");
    // dbgs() << "eventRecord: " << *eventRecord << "\n";

    // Insert cudaStreamWaitEvent
    builder.SetInsertPoint(dstInsertPt);
    loadEvent = builder.CreateLoad(eventStructPtTy, eventAlloc);
    loadStream = builder.CreateLoad(streamStructPtTy, dstStream);
    auto constZero = ConstantInt::get(Type::getInt32Ty(context), 0);
    auto waitEvent = builder.CreateCall(cudaStreamWaitEventFunc, {loadStream, loadEvent, constZero}, "newEventWait");
    // dbgs() << "waitEvent: " << *waitEvent << "\n";

    dummyAlloc->eraseFromParent();
    return eventAlloc;
}

void StreamGraph::printStream() {
    // dbgs() << "number of stream:" << getNumStream() << "\n";
    for (auto pair : IDStreamMap) {
        // dbgs() << "stream " << pair.first << ": " << *(pair.second) << "\n";
    }
}
void StreamGraph::printStreamGraph() {
    // dbgs() << "-----------------------------------------------------------------------------------\n";
    // dbgs() << "stream graph:\n";
    for (int i = 0; i < getNumStream(); i ++) {
        // dbgs() << "stream " << i << ": \n";
        for (auto node : graph[i]) {
            if (node->call) {
                // dbgs() << "  [Call]" << *(node->call) << ", kind: " << node->kind << "\n";
            }
            else {
                // dbgs() << "  [Call] streamGraphMemNode without call, kind: " << node->kind << "\n";
            }
            if (node->kind == CUDA_MEMNODE) {
                StreamGraphMemNode *memNode = dyn_cast<StreamGraphMemNode>(node);
                if (memNode->memoryObj) {
                    const char *kindStr = nullptr;
                    CallInst *memoryCall = nullptr;
                    if (node->call) {
                        kindStr = (memNode->memcpyKind == cudaMemcpyHostToDevice ? "cudaMemcpyAsync (H2D)" : "cudaMemcpyAsync (D2H)");
                        memoryCall = (memNode->memcpyKind == cudaMemcpyHostToDevice ? 
                            memNode->memoryObj->getMalloc() : memNode->memoryObj->getFree());
                    }
                    else {
                        kindStr = (memNode->memcpyKind == cudaMemcpyHostToDevice ? "cudaMallocAsync" : "cudaFreeAsync");
                        memoryCall = (memNode->memcpyKind == cudaMemcpyHostToDevice ? 
                            memNode->memoryObj->getMalloc() : memNode->memoryObj->getFree());
                    }
                    assert(kindStr && memoryCall);
                    // dbgs() << "  memory operation kind: " << kindStr << "\n";
                    // dbgs() << "  memory object: " << *memoryCall << "\n";
                }
            }
            // dbgs() << "  out edges: " << node->outEdges.size() << " edges\n";
            for (auto edge : node->outEdges) {
                assert(edge != nullptr && "edge should not be nullptr");
                assert(edge->dst != nullptr && "dst node should not be nullptr");
                edge->printDst();
            }
            // dbgs() << "\n";
        }
        // dbgs() << "-----------------------------------------------------------------------------------\n";
    }
    // dbgs() << "finish printing stream graph\n";
}
void StreamGraph::printStreamEdge() {
    // dbgs() << "stream edge:\n";
    for (auto pair : streamEdgeMap) {
        StreamGraphNode *src = pair.second->src;
        StreamGraphNode *dst = pair.second->dst;
        // dbgs() << "  src: " << *(src->call) << "\n  dst: " << *(dst->call) << "\n";
    }
}

GlobalVariable *StreamGraph::retriveStream(Value *val) {
    Instruction *instr = dyn_cast<Instruction>(val);
    GlobalVariable *stream = nullptr;
    if (getNumStream() == 0) {
        return nullptr;
    }
    if (instr == nullptr) {
        assert(false && "GlobalVariable cuStream is not found");
        return nullptr;
    }

    if (auto bitcast = dyn_cast<BitCastInst>(instr)) {
        instr = dyn_cast<Instruction>(bitcast->getOperand(0));
    }
    if (auto load = dyn_cast<LoadInst>(instr)) {
        stream = dyn_cast<GlobalVariable>(load->getOperand(0));
    }
    else {
        // dbgs() << *instr << "\n";
        assert(false && "unknown instruction when finding cuStream");
    }
    return stream;
}

AllocaInst *StreamGraph::retriveEvent(Value *val) {
    Instruction *instr = dyn_cast<Instruction>(val);
    AllocaInst *event = nullptr;
    if (instr == nullptr) {
        assert(false && "AllocaInst cuEvent is not found");
        return nullptr;
    }

    if (auto bitcast = dyn_cast<BitCastInst>(instr)) {
        instr = dyn_cast<Instruction>(bitcast->getOperand(0));
    }
    if (auto alloc = dyn_cast<AllocaInst>(instr)) {
        event = alloc;
    }
    else if (auto load = dyn_cast<LoadInst>(instr)) {
        event = dyn_cast<AllocaInst>(load->getOperand(0));
    }
    else {
        // dbgs() << *instr << "\n";
        assert(false && "unknown instruction when finding cuEvent");
    }
    return event;
}

// Generate dot file for stream graph
void StreamGraph::generateStreamGraphDotFile(std::string filename) {
    // dbgs() << "Generate dot file for stream graph\n";
    std::ofstream dotFile;
    dotFile.open(filename);
    dotFile << "digraph stream_graph {\n";
    
    std::map<StreamGraphNode*, int> nodeIDMap;
    int nodeID = 0;
    
    // Generate node ID
    for (int i = 0; i < getNumStream(); i++) {
        for (auto node : graph[i]) {
            nodeIDMap[node] = nodeID++;
        }
    }
    
    // Generate nodes information
    for (int i = 0; i < getNumStream(); i++) {
        for (auto node : graph[i]) {
            int srcID = nodeIDMap[node];
            if (node->kind != CUDA_MEMNODE) {
                std::string callName;
                callName = node->call->getCalledFunction()->getName().str();
                llvm::raw_string_ostream rso(callName);
                // node->call->print(rso);
                dotFile << "  " << srcID << " [label=\"" << rso.str() << "\", color=\"blue\"];\n";
            }
            else {
                StreamGraphMemNode *memNode = dyn_cast<StreamGraphMemNode>(node);
                assert(memNode && "Memory node is nullptr");
                MemoryObj *memObj = memNode->memoryObj;
                CallInst *call = node->call;
                // call may be nullptr
                assert(memObj && "Memory object or call instruction is nullptr");
                assert(isa<ConstantInt>(memObj->getSize()) && "Memory size is not constant");
                ConstantInt *constSize = dyn_cast<ConstantInt>(memObj->getSize());
                std::string memoryOp, deltaMem, color;
                assert(constSize && "Memory size is not constant");
                if (call) {
                    memoryOp = (memNode->memcpyKind == cudaMemcpyHostToDevice ? 
                        "cudaMemcpyAsync(H2D)" : "cudaMemcpyAsync(D2H)");
                }
                else {
                    memoryOp = (memNode->memcpyKind == cudaMemcpyHostToDevice ? 
                        "cudaMallocAsync" : "cudaFreeAsync");
                }
                if (memNode->memNodeKind == CUDA_MALLOC || memNode->memNodeKind == CUDA_MALLOC_MEMCPY) {
                    deltaMem = "(+" + std::to_string(constSize->getSExtValue()) + " B)";
                    color = "green";
                }
                else if (memNode->memNodeKind == CUDA_FREE || memNode->memNodeKind == CUDA_MEMCPY_FREE) {
                    deltaMem = "(-" + std::to_string(constSize->getSExtValue()) + " B)";
                    color = "red";
                }
                else {
                    deltaMem = "(0 B)";
                    color = "black";
                }
                dotFile << "  " << srcID << " [label=\"" << memoryOp << deltaMem << "\"" << ", color=\"" << color << "\"];\n";
            }
        }
    }

    // Generate edges information
    for (int i = 0; i < getNumStream(); i++) {
        for (auto node : graph[i]) {
            int srcID = nodeIDMap[node];
            for (auto edge : node->outEdges) {
                assert(edge != nullptr && "edge should not be nullptr");
                assert(edge->dst != nullptr && "dst node should not be nullptr");
                int dstID = nodeIDMap[edge->dst];
                std::string edgeLabel;
                int srcStreamID = getStreamID(node->stream);
                int dstStreamID = getStreamID(edge->dst->stream);
                if (srcStreamID == dstStreamID) {
                    edgeLabel = std::to_string(srcStreamID);
                }
                else {
                    edgeLabel = std::to_string(srcStreamID) + "->" + std::to_string(dstStreamID);
                }
                dotFile << "  " << srcID << " -> " << dstID << "[label=\"" << edgeLabel << "\"];\n";
            }
        }
    }
    
    dotFile << "}\n";
    dotFile.close();
    // dbgs() << "Finish generating dot file for stream graph\n";
}

// Generate dot file for memory graph
void MemoryGraph::generateMemoryGraphDotFile(std::string filename) {
    // dbgs() << "Generate dot file for memory graph\n";
    std::ofstream dotFile;
    dotFile.open(filename);
    dotFile << "digraph memory_graph {\n";

    // Generate nodes information
    for (int i = 0; i < numNode; i ++) {
        std::string color;
        if (ConstantInt *weightVal = dyn_cast<ConstantInt>(weights[i])) {
            int64_t weight = weightVal->getSExtValue();
            color = (weight > 0 ? "green" : weight == 0 ? "black" : "red");
            dotFile << "  " << i << " [label=\"" << weight << " B\"" << ", color=\"" << color << "\"];\n";
        }
        else {
            color = "black";
            assert(isa<Instruction>(weights[i]) && "Memory size should be an instruction when it is not a constant");
            Instruction *weightInstr = dyn_cast<Instruction>(weights[i]);
            std::string weightInstrStr;
            raw_string_ostream rso(weightInstrStr);
            weightInstr->print(rso);
            rso.flush();
            dotFile << "  " << i << " [label=\"" << weightInstrStr << " B\"" << ", color=\"" << color << "\"];\n";
        }
    }

    // Generate edges information
    for (int i = 0; i < numNode; i ++) {
        for (int j = 0; j < numNode; j ++) {
            if (i != j && adjMatrix[i][j]) {
                std::string edgeLabel;
                if (streamIDs[i] == streamIDs[j]) {
                    edgeLabel = std::to_string(streamIDs[i]);
                }
                else {
                    edgeLabel = std::to_string(streamIDs[i]) + 
                        "->" + std::to_string(streamIDs[j]);
                }
                dotFile << "  " << i << " -> " << j << " [label=\"" << edgeLabel << "\"];\n";
            }
        }
    }

    dotFile << "}\n";
    dotFile.close();
    // dbgs() << "Finish generating dot file for memory graph\n";
}

// Find node ID of previous memory node, return -1 if not found
int StreamGraph::findLastMemoryNodeID(int idx, const std::vector<std::vector<bool> > &memoryNodeVec) {
    int curIdx = 0;
    int streamID = 0;
    int numStream = memoryNodeVec.size();
    while (streamID < numStream && idx >= curIdx + memoryNodeVec[streamID].size()) {
        curIdx += memoryNodeVec[streamID].size();
        streamID ++;
    }
    if (streamID < numStream) {
        for (int i = idx - curIdx; i >= 0; i --) {
            if (memoryNodeVec[streamID][i]) {
                return curIdx + i;
            }
        }
    }
    return -1;
}

// Find node ID of next memory node, return -1 if not found
int StreamGraph::findNextMemoryNodeID(int idx, const std::vector<std::vector<bool> > &memoryNodeVec) {
    int curIdx = 0;
    int streamID = 0;
    int numStream = memoryNodeVec.size();
    while (streamID < numStream && idx >= curIdx + memoryNodeVec[streamID].size()) {
        curIdx += memoryNodeVec[streamID].size();
        streamID ++;
    }
    if (streamID < numStream) {
        for (int i = idx - curIdx; i < memoryNodeVec[streamID].size(); i ++) {
            if (memoryNodeVec[streamID][i]) {
                return curIdx + i;
            }
        }
    }
    return -1;
}

// Return index of given node in sequence graph
int StreamGraph::getSequenceGraphIndex(StreamGraphNode *node) {
    for (int i = 0; i < sequenceGraph.size(); i ++) {
        if (sequenceGraph[i] == node) {
            return i;
        }
    }
    assert(false && "Node is not found in sequence graph");
}

// Build memory graph from stream graph
// Assume that sizes of memory object are constant so that we can calculate the maximum memory in compile time
MemoryGraph StreamGraph::toMemoryGraph() {
    int numNode = 0;
    int numEdge = 0;
    int numStream = getNumStream();
    int numMemoryNode = 0;
    // std::vector<int64_t> weights;
    std::vector<Value *> weights;
    std::vector<int> streamIDs, src, dst;
    std::map<StreamGraphNode *, int> memNodeIDMap;
    std::map<StreamGraphNode *, int> totalNodeIDMap;
    std::map<int, StreamGraphNode *> totalIDNodeMap;
    
    // Allocate arrays for memory graph
    for (int i = 0; i < numStream; i ++) {
        for (auto node : graph[i]) {
            if (node->kind == CUDA_MEMNODE) {
                numMemoryNode ++;
            }
        }
    }
    numNode = numMemoryNode;
    streamIDs.resize(numNode);
    weights.resize(numNode);

    // Build node set of memory graph
    int memoryNodeCnt = 0;
    int totalNodeCnt = 0;
    std::vector<std::vector<bool> > memoryNodeVec(numStream, std::vector<bool>());
    for (int i = 0; i < numStream; i ++) {
        for (auto node : graph[i]) {
            totalNodeIDMap[node] = totalNodeCnt;
            totalIDNodeMap[totalNodeCnt] = node;
            memoryNodeVec[i].push_back(node->kind == CUDA_MEMNODE);
            totalNodeCnt ++;
            if (node->kind != CUDA_MEMNODE) {
                continue;
            }

            // Generate node for memory graph if StreamGraphNode is a memory node
            StreamGraphMemNode *memNode = dyn_cast<StreamGraphMemNode>(node);
            assert(memNode && "Memory node is nullptr");
            MemoryObj *memObj = memNode->memoryObj;
            CallInst *call = node->call;
            assert(memObj && "Memory object or call instruction is nullptr");
            assert(isa<IntegerType>(memObj->getSize()->getType()) && "Memory size is not integer");
            if (memNode->memNodeKind == CUDA_MALLOC || memNode->memNodeKind == CUDA_MALLOC_MEMCPY) {
                streamIDs[memoryNodeCnt] = i;
                // weights[memoryNodeCnt] = constSize->getSExtValue();
                weights[memoryNodeCnt] = memObj->getSize();
                memNodeIDMap[node] = memoryNodeCnt;
                memoryNodeCnt ++;
            }
            else if (memNode->memNodeKind == CUDA_FREE || memNode->memNodeKind == CUDA_MEMCPY_FREE) {
                Value *sizeVal = memObj->getSize();
                streamIDs[memoryNodeCnt] = i;
                if (ConstantInt *constSize = dyn_cast<ConstantInt>(sizeVal)) {
                    weights[memoryNodeCnt] = ConstantInt::get(sizeVal->getType(), -constSize->getSExtValue());
                }
                else {
                    assert(isa<Instruction>(sizeVal) && "Memory size should be an instruction when it is not a constant");
                    Instruction *sizeInstr = dyn_cast<Instruction>(sizeVal);
                    weights[memoryNodeCnt] = BinaryOperator::CreateNeg(memObj->getSize(), "", sizeInstr->getNextNonDebugInstruction());
                }
                memNodeIDMap[node] = memoryNodeCnt;
                memoryNodeCnt ++;
            }
            else {
                continue;
            }
        }
    }

    // Build edge set of memory graph
    // find the predecessors and successors of each node
    for (int i = 0; i < numStream; i ++) {
        for (auto node : graph[i]) {
            for (auto edge : node->outEdges) {
                int srcID = findLastMemoryNodeID(totalNodeIDMap[node], memoryNodeVec);
                int dstID = findNextMemoryNodeID(totalNodeIDMap[edge->dst], memoryNodeVec);
                if (srcID != -1 && dstID != -1) {
                    StreamGraphNode *srcNode = totalIDNodeMap[srcID];
                    StreamGraphNode *dstNode = totalIDNodeMap[dstID];
                    src.push_back(memNodeIDMap[srcNode]);
                    dst.push_back(memNodeIDMap[dstNode]);
                }
            }
        }
    }
    numEdge = src.size();

    // Assign values to pointers
    // dbgs() << "numNode: " << numNode << ", numEdge: " << numEdge << ", numStream: " << numStream << "\n";
    MemoryGraph memoryGraph(streamIDs, weights, src, dst, numNode, numEdge, numStream);
    return memoryGraph;
}

void MemoryGraph::initialize(std::vector<int> src, std::vector<int> dst, int numNodeIn, int numEdgeIn, int numStreamIn) {
    this->numNode = numNodeIn;
    this->numEdge = numEdgeIn;
    this->numStream = numStreamIn;
    originalMaxMemory = 0;
    adjMatrix.resize(numNode, std::vector<bool>(numNode, false));
    reachMatrix.resize(numNode, std::vector<bool>(numNode, false));
    weightLists.resize(numStream, std::vector<Value *>());
    // dbgs() << "numNode: " << numNode << ", numEdge: " << numEdge << ", numStream: " << numStream << "\n";

    // Cummulate weights
    for (int i = 0; i < numNode; i ++) {
        int id = streamIDs[i];
        weightLists[id].push_back(weights[i]);
    }

    // Build adjacent matrix
    for (int i = 0; i < numEdge; i ++) {
        reachMatrix[src[i]][dst[i]] = true;
        adjMatrix[src[i]][dst[i]] = true;
    }
    // Build reachable matrix for dependency checking (Warshall Algorithm)
    for (int i = 0; i < numNode; i ++) {
        for (int j = 0; j < numNode; j ++) {
            if (reachMatrix[i][j]) {
                for (int k = 0; k < numNode; k ++) {
                    reachMatrix[i][k] = (reachMatrix[i][k] || reachMatrix[j][k]);
                }
            }
        }
    }
}

bool MemoryGraph::checkDependency(const std::vector<int> &nodeList) {
    for (int i = 0; i < nodeList.size(); i ++) {
        for (int j = i + 1; j < nodeList.size(); j ++ ) {
            if (reachMatrix[i][j] || reachMatrix[j][i]) {
                return true;
            }
        }
    }
    return false;
}

uint64_t MemoryGraph::getMaxMemory() {
    assert(false && "Get max memory in runtime system");
}

// Transform memory graph to LLVM arrays and pointers
// `uint64_t **weightList2D` contains the whole memory graph
// `uint64_t *graphSize1D` contains the size of each stream
// `uint64_t numStreamConst` indicates the number of streams
SmallVector<Value *, 3> MemoryGraph::toLLVMArray(Instruction *insertPoint) {
    assert(insertPoint && "Insert pointer should not be nullptr");
    IRBuilder<> builder(insertPoint);
    Type *int64Ty = builder.getInt64Ty();
    Type *int64PtrTy = int64Ty->getPointerTo();
    Constant *numStreamConst = ConstantInt::get(int64Ty, numStream);

    AllocaInst *weightList2D = builder.CreateAlloca(int64PtrTy, numStreamConst, "weightLists");
    AllocaInst *graphSize1D = builder.CreateAlloca(int64Ty, numStreamConst, "graphSize1D");
    for (int i = 0; i < numStream; i ++) {
        Value *streamIdx = ConstantInt::get(int64Ty, i);
        Constant *listSize = ConstantInt::get(int64Ty, weightLists[i].size());
        if (weightLists[i].size()) {
            AllocaInst *weightList1D = builder.CreateAlloca(int64Ty, listSize, "weightList");
            for (int j = 0; j < weightLists[i].size(); j ++) {
                Value *idx = ConstantInt::get(int64Ty, j);
                Value *gep = builder.CreateGEP(int64Ty, weightList1D, idx);
                builder.CreateStore(weightLists[i][j], gep);
            }
            Value *gep = builder.CreateGEP(int64PtrTy, weightList2D, streamIdx);
            builder.CreateStore(weightList1D, gep);
        }
        Value *gep = builder.CreateGEP(int64Ty, graphSize1D, streamIdx);
        builder.CreateStore(listSize, gep);
    }

    return SmallVector<Value *, 3>{weightList2D, graphSize1D, numStreamConst};
}

KernelInvoke *AnalysisResult::getKernelInvoke(CallInst *call) {
    if (kernelInvokeMap.find(call) == kernelInvokeMap.end()) {
        return nullptr;
    }
    return kernelInvokeMap[call];
}
