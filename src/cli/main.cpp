#include "Compiler.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input .mlir file>"),
    llvm::cl::Required);

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "Sandy Compiler\n");

  sandy::Compiler compiler;
  auto module = compiler.loadModule(inputFilename);
  if (!module) {
    llvm::errs() << "Failed to parse: " << inputFilename << "\n";
    return 1;
  }

  compiler.dump(*module);
  return 0;
}
