#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

#include <iostream>

namespace l0op {
    OP_TYPE_REGISTER(PermuteCustom);

    aclTensor* PermuteCustom(const aclTensor* input, const aclIntArray *perm, aclOpExecutor* executor) {
        L0_DFX(PermuteCustom, input);

        auto output = executor->AllocTensor(input->GetDataType(), input->GetStorageFormat(), input->GetOriginalFormat());

        auto ret = INFER_SHAPE(PermuteCustom, OP_INPUT(input), OP_ATTR(perm), OP_OUTPUT(output));

        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PermuteCustom InferShape failed.");
            return nullptr;
        }

        ADD_TO_LAUNCHER_LIST_AICORE(PermuteCustom, OP_INPUT(input), OP_ATTR(perm), OP_OUTPUT(output));

        return output;
    }
} // namespace l0op