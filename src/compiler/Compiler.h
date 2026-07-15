#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include <string>

namespace sandy {

class Compiler {
public:
  Compiler();

  mlir::OwningOpRef<mlir::ModuleOp> loadModule(llvm::StringRef filename);
  void dump(mlir::ModuleOp module);

private:
  mlir::MLIRContext context;
};

}  // namespace sandy
