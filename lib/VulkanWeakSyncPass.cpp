// lib/VulkanWeakSyncPass.cpp
#include "VulkanWeakSyncPass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/CommandLine.h"
#include <vector>

using namespace llvm;

static cl::opt<std::string> SyncMode(
    "vk-sync-mode",
    cl::desc("Choose weak sync mechanism: 'non-private', 'v3', 'v1', 'v2'"),
    cl::init("v3"));

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

  auto TagInst = [&](Instruction *I, bool isClosest = false) {
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
  auto nonPrivatePass = [&]() {
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
                Name.contains("spirv.op.224") ||
                Name.contains("spirv.op.225")) {
              isAcquire = true;
              isRelease = true;
            }
            // Catch clspv lowered SPIR-V atomics (227 through 242)
            else if (Name.contains("spirv.op.22") ||
                     Name.contains("spirv.op.23") ||
                     Name.contains("spirv.op.24")) {
              if (Call->arg_size() > 3) {
                if (auto *SemC =
                        dyn_cast<ConstantInt>(Call->getArgOperand(3))) {
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
              llvm::errs() << "[VK SYNC] Hit previous sync point, stopping "
                              "backward scan: "
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
                llvm::errs()
                    << "[VK SYNC] Found Store in Predecessor: " << *Store
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
            for (auto fwdIt = CurrBB->begin(); fwdIt != CurrBB->end();
                 ++fwdIt) {
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
  };

  auto fix3Pass = [&]() {
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
                Name.contains("spirv.op.224") ||
                Name.contains("spirv.op.225")) {
              isAcquire = true;
              isRelease = true;
            } else if (Name.contains("spirv.op.22") ||
                       Name.contains("spirv.op.23") ||
                       Name.contains("spirv.op.24")) {
              if (Call->arg_size() > 3) {
                if (auto *SemC =
                        dyn_cast<ConstantInt>(Call->getArgOperand(3))) {
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
            for (auto fwdIt = CurrBB->begin(); fwdIt != CurrBB->end();
                 ++fwdIt) {
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
                // If this store points to any of our known synced
                // variables, tag it
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
                // If this load points to any of our known synced variables,
                // tag it
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
  };

  auto fix1Pass = [&]() {
    std::vector<MemoryLocation> WeakLocs;
    errs() << "in fix1Pass";
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
                Name.contains("spirv.op.224") ||
                Name.contains("spirv.op.225")) {
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
  };
  errs() << "SyncMode: " << SyncMode << "\n";
  if (SyncMode == "non-private") {
    nonPrivatePass();
  } else if (SyncMode == "v3") {
    fix3Pass();
  } else if (SyncMode == "v1" || SyncMode == "v2") {

    fix1Pass();
  }

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

} // namespace clspv