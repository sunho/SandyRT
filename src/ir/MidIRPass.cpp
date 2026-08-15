#include "MidIRPass.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

namespace sandy::ir::mid_ir {

namespace {

bool int_attr(const AttrMap& attrs, const char* name) {
    auto it = attrs.find(name);
    return it != attrs.end() && it->second.kind == AttrValue::Int && it->second.intVal != 0;
}

void set_int_attr(Op& op, const char* name, bool value) {
    if (value) {
        op.attrs[name] = AttrValue::make_int(1);
    } else {
        op.attrs.erase(name);
    }
}

bool is_graph_output(const Graph& graph, const Value* value) {
    const auto& outputs = graph.outputs();
    return std::find(outputs.begin(), outputs.end(), value) != outputs.end();
}

class FuseTransposeIntoMatMulPass final : public Pass {
public:
    const char* name() const override { return "fuse-transpose-into-matmul"; }

    Result<PassResult> run(Graph& graph) override {
        bool changed = false;
        auto ops = graph.entry()->ops;
        for (auto* op : ops) {
            if (!op || op->parent == nullptr || op->kind != OpKind::MatMul)
                continue;

            changed = fuseOperand(graph, *op, 0, "transpose_lhs") || changed;
            changed = fuseOperand(graph, *op, 1, "transpose_rhs") || changed;
        }
        return PassResult{changed};
    }

private:
    static bool fuseOperand(Graph& graph, Op& matmul, int operandIndex, const char* attrName) {
        if (operandIndex < 0 || static_cast<size_t>(operandIndex) >= matmul.operands.size())
            return false;

        auto* transposedValue = matmul.operands[static_cast<size_t>(operandIndex)];
        if (!transposedValue || !transposedValue->def || transposedValue->def->kind != OpKind::Transpose)
            return false;

        auto* transposeOp = transposedValue->def;
        if (transposeOp->operands.size() != 1 || transposeOp->results.size() != 1)
            return false;

        bool oldTranspose = int_attr(matmul.attrs, attrName);
        auto* originalValue = transposeOp->operands[0];
        graph.replaceOperand(&matmul, operandIndex, originalValue);
        set_int_attr(matmul, attrName, !oldTranspose);

        if (transposedValue->uses.empty() && !is_graph_output(graph, transposedValue))
            graph.eraseOp(transposeOp);
        return true;
    }
};

class DeadCodeEliminationPass final : public Pass {
public:
    const char* name() const override { return "dead-code-elimination"; }

    Result<PassResult> run(Graph& graph) override {
        bool changed = false;
        bool localChanged = true;
        while (localChanged) {
            localChanged = false;
            auto ops = graph.entry()->ops;
            for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
                auto* op = *it;
                if (!op || op->parent == nullptr)
                    continue;
                if (!isDead(graph, *op))
                    continue;
                if (graph.eraseOp(op)) {
                    changed = true;
                    localChanged = true;
                }
            }
        }
        return PassResult{changed};
    }

private:
    static bool isDead(const Graph& graph, const Op& op) {
        if (op.results.empty())
            return false;
        for (auto* result : op.results) {
            if (!result->uses.empty() || is_graph_output(graph, result))
                return false;
        }
        return true;
    }
};

} // namespace

void PassManager::add(std::unique_ptr<Pass> pass) {
    passes_.push_back(std::move(pass));
}

Result<PassResult> PassManager::run(Graph& graph) {
    bool changed = false;
    for (auto& pass : passes_) {
        auto result = pass->run(graph);
        if (!result)
            return make_error(result.error());
        changed = result->changed || changed;
    }
    return PassResult{changed};
}

std::unique_ptr<Pass> createFuseTransposeIntoMatMulPass() {
    return std::make_unique<FuseTransposeIntoMatMulPass>();
}

std::unique_ptr<Pass> createDeadCodeEliminationPass() {
    return std::make_unique<DeadCodeEliminationPass>();
}

} // namespace sandy::ir::mid_ir
