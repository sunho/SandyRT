#include "Compiler.h"
#include "SafeTensorWeights.h"

#include <cstdio>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: sandy <model.sandy.go> [weights.safetensors]\n");
        return 1;
    }

    sandy::Compiler compiler;
    auto graph = compiler.load_sandygo(argv[1]);
    graph.dump();

    if (argc >= 3) {
        auto weights = sandy::weight::EagerSafeTensorWeights::load(argv[2]);
        auto result = compiler.materialize_mid_ir(graph, *weights);
        if (!result) {
            fprintf(stderr, "materialize error: %s\n", result.error().c_str());
            return 1;
        }
        auto midGraph = result.take();
        midGraph->dump();
    }

    return 0;
}
