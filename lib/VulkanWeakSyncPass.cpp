// lib/VulkanWeakSyncPass.cpp
#include "VulkanWeakSyncPass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include <vector>

using namespace llvm;

namespace clspv {

PreservedAnalyses VulkanWeakSyncPass::run(Function &F,
                                          FunctionAnalysisManager &FAM) {
  AAResults &AA = FAM.getResult<AAManager>(F);
  LLVMContext &Ctx = F.getContext();

  auto TagInst = [&](Instruction *I) {
    if (!I->getMetadata("vk.weak")) {
      MDNode *Node = MDNode::get(Ctx, MDString::get(Ctx, "vk.weak"));
      I->setMetadata("vk.weak", Node);
      llvm::errs() << "[VK SYNC] >>> SUCCESS: Tagged shared memory instance: "
                   << *I << "\n";
    }
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

  // List of all memory locations that participate in synchronization
  std::vector<MemoryLocation> WeakLocs;

  // Find the Weak Locations
  for (auto &BB : F) {
    for (auto it = BB.begin(); it != BB.end(); ++it) {
      Instruction *Inst = &*it;
      bool isAcquire = false;
      bool isRelease = false;

      // Identify Sync Points
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
            if (Call->arg_size() > 3) {
              if (auto *SemC = dyn_cast<ConstantInt>(Call->getArgOperand(3))) {
                uint32_t sem = SemC->getZExtValue();
                if ((sem & 0x2) || (sem & 0x8) || (sem & 0x10))
                  isAcquire = true;
                if ((sem & 0x4) || (sem & 0x8) || (sem & 0x10))
                  isRelease = true;
              }
            }
          }
        }
      }

      // collect synced memory locations
      if (isRelease) {
        SmallVector<BasicBlock *, 8> Worklist;
        SmallPtrSet<BasicBlock *, 8> Visited;
        BasicBlock *StartBB = Inst->getParent();
        bool hitSync = false;

        for (auto backIt = BasicBlock::reverse_iterator(it);
             backIt != StartBB->rend(); ++backIt) {
          Instruction *Prev = &*backIt;
          if (isSyncPoint(Prev)) {
            hitSync = true;
            break;
          }

          if (auto *Store = dyn_cast<StoreInst>(Prev)) {
            unsigned AS = Store->getPointerAddressSpace();
            if (AS == 1 || AS == 3) { // Only Global/Local
              WeakLocs.push_back(MemoryLocation::get(Store));
            }
          }
        }

        if (!hitSync) {
          for (BasicBlock *Pred : predecessors(StartBB))
            Worklist.push_back(Pred);
        }

        while (!Worklist.empty()) {
          BasicBlock *CurrBB = Worklist.pop_back_val();
          if (!Visited.insert(CurrBB).second)
            continue;

          hitSync = false;
          for (auto backIt = CurrBB->rbegin(); backIt != CurrBB->rend();
               ++backIt) {
            Instruction *Prev = &*backIt;
            if (isSyncPoint(Prev)) {
              hitSync = true;
              break;
            }

            if (auto *Store = dyn_cast<StoreInst>(Prev)) {
              unsigned AS = Store->getPointerAddressSpace();
              if (AS == 1 || AS == 3) {
                WeakLocs.push_back(MemoryLocation::get(Store));
              }
            }
          }
          if (!hitSync) {
            for (BasicBlock *Pred : predecessors(CurrBB))
              Worklist.push_back(Pred);
          }
        }
      }

      // Collect synced memory locations
      if (isAcquire) {
        SmallVector<BasicBlock *, 8> Worklist;
        SmallPtrSet<BasicBlock *, 8> Visited;
        BasicBlock *StartBB = Inst->getParent();
        bool hitSync = false;

        for (auto fwdIt = std::next(BasicBlock::iterator(Inst));
             fwdIt != StartBB->end(); ++fwdIt) {
          Instruction *Next = &*fwdIt;
          if (isSyncPoint(Next)) {
            hitSync = true;
            break;
          }

          if (auto *Load = dyn_cast<LoadInst>(Next)) {
            unsigned AS = Load->getPointerAddressSpace();
            if (AS == 1 || AS == 3) {
              WeakLocs.push_back(MemoryLocation::get(Load));
            }
          }
        }

        if (!hitSync) {
          for (BasicBlock *Succ : successors(StartBB))
            Worklist.push_back(Succ);
        }

        while (!Worklist.empty()) {
          BasicBlock *CurrBB = Worklist.pop_back_val();
          if (!Visited.insert(CurrBB).second)
            continue;

          hitSync = false;
          for (auto fwdIt = CurrBB->begin(); fwdIt != CurrBB->end(); ++fwdIt) {
            Instruction *Next = &*fwdIt;
            if (isSyncPoint(Next)) {
              hitSync = true;
              break;
            }

            if (auto *Load = dyn_cast<LoadInst>(Next)) {
              unsigned AS = Load->getPointerAddressSpace();
              if (AS == 1 || AS == 3) {
                WeakLocs.push_back(MemoryLocation::get(Load));
              }
            }
          }
          if (!hitSync) {
            for (BasicBlock *Succ : successors(CurrBB))
              Worklist.push_back(Succ);
          }
        }
      }
    }
  }

  // Tag all global instances
  if (!WeakLocs.empty()) {
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *Store = dyn_cast<StoreInst>(&I)) {
          unsigned AS = Store->getPointerAddressSpace();
          if (AS == 1 || AS == 3) {
            MemoryLocation Loc = MemoryLocation::get(Store);
            for (auto &WeakLoc : WeakLocs) {
              // If this store points to any of our known synced variables, tag it
              if (AA.alias(Loc, WeakLoc) == AliasResult::MustAlias) {
                TagInst(Store);
                break;
              }
            }
          }
        } else if (auto *Load = dyn_cast<LoadInst>(&I)) {
          unsigned AS = Load->getPointerAddressSpace();
          if (AS == 1 || AS == 3) {
            MemoryLocation Loc = MemoryLocation::get(Load);
            for (auto &WeakLoc : WeakLocs) {
              // If this load points to any of our known synced variables, tag it
              if (AA.alias(Loc, WeakLoc) == AliasResult::MustAlias) {
                TagInst(Load);
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