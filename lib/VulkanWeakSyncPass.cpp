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
      llvm::errs() << "[VK SYNC] Tagged instruction: " << *I
                   << "\n";
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

  for (auto &BB : F) {
    for (auto it = BB.begin(); it != BB.end(); ++it) {
      Instruction *Inst = &*it;
      bool isAcquire = false;
      bool isRelease = false;

      // Identify sync points
      if (auto *Call = dyn_cast<CallInst>(Inst)) {
        Function *CalledFunc = Call->getCalledFunction();
        if (CalledFunc && CalledFunc->isDeclaration()) {
          StringRef Name = CalledFunc->getName();

          if (Name.contains("barrier") || Name.contains("fence") ||
              Name.contains("spirv.op.224") || Name.contains("spirv.op.225")) {
            isAcquire = true;
            isRelease = true;
          }
          // Catch clspv lowered SPIR-V atomics (227 through 242)
          else if (Name.contains("spirv.op.22") ||
                   Name.contains("spirv.op.23") ||
                   Name.contains("spirv.op.24")) {
            if (Call->arg_size() > 3) {
              if (auto *SemC = dyn_cast<ConstantInt>(Call->getArgOperand(3))) {
                uint32_t sem = SemC->getZExtValue();
                if ((sem & 0x2) || (sem & 0x8) || (sem & 0x10))
                  isAcquire = true;
                if ((sem & 0x4) || (sem & 0x8) || (sem & 0x10))
                  isRelease = true;

                if (isAcquire || isRelease) {
                  llvm::errs()
                      << "\n[VK SYNC] FOUND SPIR-V ATOMIC: " << Name << "\n";
                  llvm::errs() << "[VK SYNC] Semantics: " << sem
                               << " -> Acquire: " << isAcquire
                               << ", Release: " << isRelease << "\n";
                }
              }
            }
          }
        }
      }

      // Handle Release
      if (isRelease) {
        llvm::errs() << "[VK SYNC] Scanning BACKWARDS for Release...\n";
        std::vector<MemoryLocation> TaggedLocs;
        SmallVector<BasicBlock *, 8> Worklist;
        SmallPtrSet<BasicBlock *, 8> Visited;

        BasicBlock *StartBB = Inst->getParent();
        bool hitSync = false;

        for (auto backIt = BasicBlock::reverse_iterator(it);
             backIt != StartBB->rend(); ++backIt) {
          Instruction *Prev = &*backIt;

          if (isSyncPoint(Prev)) {
            llvm::errs()
                << "[VK SYNC] Hit previous sync point, stopping backward scan: "
                << *Prev << "\n";
            hitSync = true;
            break;
          }

          if (auto *Store = dyn_cast<StoreInst>(Prev)) {
            llvm::errs() << "[VK SYNC] Found Store: " << *Store << "\n";
            MemoryLocation Loc = MemoryLocation::get(Store);
            bool alreadySynced = false;
            for (auto &TaggedLoc : TaggedLocs) {
              if (AA.alias(Loc, TaggedLoc) == AliasResult::MustAlias) {
                alreadySynced = true;
                break;
              }
            }
            if (!alreadySynced) {
              TagInst(Store);
              TaggedLocs.push_back(Loc);
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
              llvm::errs() << "[VK SYNC] Found Store in Predecessor: " << *Store
                           << "\n";
              MemoryLocation Loc = MemoryLocation::get(Store);
              bool alreadySynced = false;
              for (auto &TaggedLoc : TaggedLocs) {
                if (AA.alias(Loc, TaggedLoc) == AliasResult::MustAlias) {
                  alreadySynced = true;
                  break;
                }
              }
              if (!alreadySynced) {
                TagInst(Store);
                TaggedLocs.push_back(Loc);
              }
            }
          }
          if (!hitSync) {
            for (BasicBlock *Pred : predecessors(CurrBB))
              Worklist.push_back(Pred);
          }
        }
      }

      // Handle Acquire 
      if (isAcquire) {
        std::vector<MemoryLocation> TaggedLocs;
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
            MemoryLocation Loc = MemoryLocation::get(Load);
            bool alreadySynced = false;
            for (auto &TaggedLoc : TaggedLocs) {
              if (AA.alias(Loc, TaggedLoc) == AliasResult::MustAlias) {
                alreadySynced = true;
                break;
              }
            }
            if (!alreadySynced) {
              TagInst(Load);
              TaggedLocs.push_back(Loc);
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
              MemoryLocation Loc = MemoryLocation::get(Load);
              bool alreadySynced = false;
              for (auto &TaggedLoc : TaggedLocs) {
                if (AA.alias(Loc, TaggedLoc) == AliasResult::MustAlias) {
                  alreadySynced = true;
                  break;
                }
              }
              if (!alreadySynced) {
                TagInst(Load);
                TaggedLocs.push_back(Loc);
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

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

} // namespace clspv