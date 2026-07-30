#include "SafeTensorWeights.h"

#include <cstdio>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: load_safetensors <file.safetensors>\n");
        return 1;
    }

    auto weights = weight::EagerSafeTensorWeights::load(argv[1]);
    auto descs = weights.get_descriptors();

    printf("%zu tensors:\n", descs.size());
    for (auto& d : descs) {
        printf("  %-30s %s  %s\n",
               d.name.c_str(),
               d.shape.str().c_str(),
               ir::dtype_name(d.dtype));
    }

    return 0;
}
