// lib/VulkanWeakSyncPass.cpp
#include "VulkanWeakSyncPass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include <string>
#include <vector>

using namespace llvm;

namespace clspv {


// SPIR-V memory semantics
constexpr uint32_t MemSemanticsAcquire = 0x2;
constexpr uint32_t MemSemanticsRelease = 0x4;
constexpr uint32_t MemSemanticsAcquireRelease = 0x8;
constexpr uint32_t MemSemanticsSequentiallyConsistent = 0x10;

// OpenCL / SPIR-V Address Spaces
constexpr unsigned AddrSpaceGlobal = 1;
constexpr unsigned AddrSpaceLocal = 3;

// Clspv internal mem operand index
constexpr unsigned BuiltinMemSemanticsArgIdx = 3;


struct BFSNode {
  BasicBlock *BB;
  std::vector<MemoryLocation> FoundLocs;
};

PreservedAnalyses VulkanWeakSyncPass::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  AAResults &AA = FAM.getResult<AAManager>(F);
  LLVMContext &Ctx = F.getContext();

  auto TagInst = [&](Instruction *I, bool isClosest) {
    std::string tagStr = isClosest ? "vk_weak_closest" : "vk_weak";
    MDNode *Existing = I->getMetadata("vk.weak");
    if (Existing) {
      auto *MDS = dyn_cast<MDString>(Existing->getOperand(0));
      if (MDS && MDS->getString().starts_with("vk_weak_closest"))
        return;
    }
    MDNode *Node = MDNode::get(Ctx, MDString::get(Ctx, tagStr));
    I->setMetadata("vk.weak", Node);
  };

  auto isSyncPoint = [](Instruction *I) {
    if (I->isAtomic())
      return true;
    if (isa<FenceInst>(I))
      return true;
    if (auto *Call = dyn_cast<CallInst>(I)) {
      Function *CalledFunc = Call->getCalledFunction();
      if (CalledFunc && CalledFunc->isDeclaration()) {
        StringRef Name = CalledFunc->getName();
        if (Name.contains("barrier") || Name.contains("fence") ||
            Name.contains("spirv.op.22") || Name.contains("spirv.op.23") ||
            Name.contains("spirv.op.24")) {
          return true;
        }
      }
    }
    return false;
  };

  std::vector<MemoryLocation> WeakLocs;

  // Discover closest instance of a weak variable to sync point
  for (auto &BB : F) {
    for (auto it = BB.begin(); it != BB.end(); ++it) {
      Instruction *Inst = &*it;
      bool isAcquire = false;
      bool isRelease = false;

      if (auto *Call = dyn_cast<CallInst>(Inst)) {
        Function *CalledFunc = Call->getCalledFunction();
        if (CalledFunc && CalledFunc->isDeclaration()) {
          StringRef Name = CalledFunc->getName();
          if (Name.contains("barrier") || Name.contains("fence") ||
              Name.contains("spirv.op.224") || Name.contains("spirv.op.225")) {
            isAcquire = true;
            isRelease = true;
          } else if (Name.contains("spirv.op.22") ||
                     Name.contains("spirv.op.23") ||
                     Name.contains("spirv.op.24")) {
            if (Call->arg_size() > BuiltinMemSemanticsArgIdx) {
              if (auto *SemC = dyn_cast<ConstantInt>(
                      Call->getArgOperand(BuiltinMemSemanticsArgIdx))) {
                uint32_t sem = SemC->getZExtValue();
                if ((sem & MemSemanticsAcquire) ||
                    (sem & MemSemanticsAcquireRelease) ||
                    (sem & MemSemanticsSequentiallyConsistent))
                  isAcquire = true;
                if ((sem & MemSemanticsRelease) ||
                    (sem & MemSemanticsAcquireRelease) ||
                    (sem & MemSemanticsSequentiallyConsistent))
                  isRelease = true;
              }
            }
          }
        }
      }

      if (!isAcquire && !isRelease)
        continue;

      // Handle release (traverse backwards)
      if (isRelease) {
        SmallVector<BFSNode, 8> Worklist;
        SmallPtrSet<BasicBlock *, 8> Visited;
        BasicBlock *StartBB = Inst->getParent();
        bool hitSync = false;
        std::vector<MemoryLocation> StartFoundLocs;

        for (auto backIt = BasicBlock::reverse_iterator(it);
             backIt != StartBB->rend(); ++backIt) {
          Instruction *Prev = &*backIt;
          if (isSyncPoint(Prev)) {
            hitSync = true;
            break;
          }

          if (auto *Store = dyn_cast<StoreInst>(Prev)) {
            unsigned AS = Store->getPointerAddressSpace();
            if (AS == AddrSpaceGlobal || AS == AddrSpaceLocal) {
              MemoryLocation Loc = MemoryLocation::get(Store);
              bool alreadyFound = false;
              for (auto &Found : StartFoundLocs) {
                if (AA.alias(Loc, Found) == AliasResult::MustAlias) {
                  alreadyFound = true;
                  break;
                }
              }
              if (!alreadyFound) {
                StartFoundLocs.push_back(Loc);
                TagInst(Store, true);
                WeakLocs.push_back(Loc);
              }
            }
          }
        }

        if (!hitSync) {
          for (BasicBlock *Pred : predecessors(StartBB))
            Worklist.push_back({Pred, StartFoundLocs});
        }

        while (!Worklist.empty()) {
          BFSNode Node = Worklist.pop_back_val();
          if (!Visited.insert(Node.BB).second)
            continue;

          hitSync = false;
          for (auto backIt = Node.BB->rbegin(); backIt != Node.BB->rend();
               ++backIt) {
            Instruction *Prev = &*backIt;
            if (isSyncPoint(Prev)) {
              hitSync = true;
              break;
            }

            if (auto *Store = dyn_cast<StoreInst>(Prev)) {
              unsigned AS = Store->getPointerAddressSpace();
              if (AS == AddrSpaceGlobal || AS == AddrSpaceLocal) {
                MemoryLocation Loc = MemoryLocation::get(Store);
                bool alreadyFound = false;
                for (auto &Found : Node.FoundLocs) {
                  if (AA.alias(Loc, Found) == AliasResult::MustAlias) {
                    alreadyFound = true;
                    break;
                  }
                }
                if (!alreadyFound) {
                  Node.FoundLocs.push_back(Loc);
                  TagInst(Store, true);
                  WeakLocs.push_back(Loc);
                }
              }
            }
          }
          if (!hitSync) {
            for (BasicBlock *Pred : predecessors(Node.BB))
              Worklist.push_back({Pred, Node.FoundLocs});
          }
        }
      }

      // Handle acquire (traverse forward)
      if (isAcquire) {
        SmallVector<BFSNode, 8> Worklist;
        SmallPtrSet<BasicBlock *, 8> Visited;
        BasicBlock *StartBB = Inst->getParent();
        bool hitSync = false;
        std::vector<MemoryLocation> StartFoundLocs;

        for (auto fwdIt = std::next(BasicBlock::iterator(Inst));
             fwdIt != StartBB->end(); ++fwdIt) {
          Instruction *Next = &*fwdIt;
          if (isSyncPoint(Next)) {
            hitSync = true;
            break;
          }

          if (auto *Load = dyn_cast<LoadInst>(Next)) {
            unsigned AS = Load->getPointerAddressSpace();
            if (AS == AddrSpaceGlobal || AS == AddrSpaceLocal) {
              MemoryLocation Loc = MemoryLocation::get(Load);
              bool alreadyFound = false;
              for (auto &Found : StartFoundLocs) {
                if (AA.alias(Loc, Found) == AliasResult::MustAlias) {
                  alreadyFound = true;
                  break;
                }
              }
              if (!alreadyFound) {
                StartFoundLocs.push_back(Loc);
                TagInst(Load, true);
                WeakLocs.push_back(Loc);
              }
            }
          }
        }

        if (!hitSync) {
          for (BasicBlock *Succ : successors(StartBB))
            Worklist.push_back({Succ, StartFoundLocs});
        }

        while (!Worklist.empty()) {
          BFSNode Node = Worklist.pop_back_val();
          if (!Visited.insert(Node.BB).second)
            continue;

          hitSync = false;
          for (auto fwdIt = Node.BB->begin(); fwdIt != Node.BB->end();
               ++fwdIt) {
            Instruction *Next = &*fwdIt;
            if (isSyncPoint(Next)) {
              hitSync = true;
              break;
            }

            if (auto *Load = dyn_cast<LoadInst>(Next)) {
              unsigned AS = Load->getPointerAddressSpace();
              if (AS == AddrSpaceGlobal || AS == AddrSpaceLocal) {
                MemoryLocation Loc = MemoryLocation::get(Load);
                bool alreadyFound = false;
                for (auto &Found : Node.FoundLocs) {
                  if (AA.alias(Loc, Found) == AliasResult::MustAlias) {
                    alreadyFound = true;
                    break;
                  }
                }
                if (!alreadyFound) {
                  Node.FoundLocs.push_back(Loc);
                  TagInst(Load, true);
                  WeakLocs.push_back(Loc);
                }
              }
            }
          }
          if (!hitSync) {
            for (BasicBlock *Succ : successors(Node.BB))
              Worklist.push_back({Succ, Node.FoundLocs});
          }
        }
      }
    }
  }

  //  Tag instructions
  if (!WeakLocs.empty()) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *Store = dyn_cast<StoreInst>(&I)) {
          unsigned AS = Store->getPointerAddressSpace();
          if (AS == AddrSpaceGlobal || AS == AddrSpaceLocal) {
            MemoryLocation Loc = MemoryLocation::get(Store);
            for (auto &WeakLoc : WeakLocs) {
              if (AA.alias(Loc, WeakLoc) == AliasResult::MustAlias) {
                TagInst(Store, false);
                break;
              }
            }
          }
        } else if (auto *Load = dyn_cast<LoadInst>(&I)) {
          unsigned AS = Load->getPointerAddressSpace();
          if (AS == AddrSpaceGlobal || AS == AddrSpaceLocal) {
            MemoryLocation Loc = MemoryLocation::get(Load);
            for (auto &WeakLoc : WeakLocs) {
              if (AA.alias(Loc, WeakLoc) == AliasResult::MustAlias) {
                TagInst(Load, false);
                break;
              }
            }
          }
        }
      }
    }
  }

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

} // namespace clspv