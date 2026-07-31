#ifndef CLSPV_LIB_VULKAN_WEAK_SYNC_PASS_H
#define CLSPV_LIB_VULKAN_WEAK_SYNC_PASS_H

#include "llvm/IR/PassManager.h"

namespace clspv {

struct VulkanWeakSyncPass : llvm::PassInfoMixin<VulkanWeakSyncPass> {
  llvm::PreservedAnalyses run(llvm::Function &F, llvm::FunctionAnalysisManager &FAM);
};

} // namespace clspv

#endif 