#include "Compiler.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "stablehlo/dialect/Register.h"

namespace sandy {

Compiler::Compiler() {
  mlir::DialectRegistry registry;
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect>();
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

mlir::OwningOpRef<mlir::ModuleOp> Compiler::loadModule(llvm::StringRef filename) {
  return mlir::parseSourceFile<mlir::ModuleOp>(filename, &context);
}

void Compiler::dump(mlir::ModuleOp module) {
  module->print(llvm::outs());
  llvm::outs() << "\n";
}

}  // namespace sandy
