#pragma once

#include "MidIR.h"
#include "Result.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sandy::ir::mid_ir {

class BuiltinLowering {
public:
    using LowerFn = std::function<Result<std::vector<Value*>>(
        Builder& builder,
        const std::vector<Value*>& operands,
        const AttrMap& attrs,
        int numResults)>;

    void add(const std::string& name, LowerFn fn);
    const LowerFn* lookup(const std::string& name) const;

    static BuiltinLowering createDefault();

private:
    std::unordered_map<std::string, LowerFn> lowerings_;
};

} // namespace sandy::ir::mid_ir
