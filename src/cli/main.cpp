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
        auto weights = weight::EagerSafeTensorWeights::load(argv[2]);
        auto midGraph = compiler.materialize_mid_ir(graph, weights);
    }

    return 0;
}
