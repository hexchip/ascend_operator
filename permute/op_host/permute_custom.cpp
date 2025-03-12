#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

#include "opdev/shape_utils.h"

#include <iostream>

namespace l0op {
    OP_TYPE_REGISTER(PermuteCustom);

    aclTensor* PermuteCustom(const aclTensor* input, const aclIntArray *perm, aclOpExecutor* executor) {
        L0_DFX(PermuteCustom, input, perm);

        auto output = executor->AllocTensor(input->GetDataType(), input->GetStorageFormat(), input->GetOriginalFormat());

        auto ret = INFER_SHAPE(PermuteCustom, OP_INPUT(input), OP_OUTPUT(output), OP_ATTR(perm));

        if (ret != ACLNN_SUCCESS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "PermuteCustom InferShape failed.");
            return nullptr;
        }

        std::cout << "output shape: " << op::ToString(output->GetStorageShape()).GetString() << std::endl;
        std::cout << "output: " << output->ToString().GetString() << std::endl;

        ADD_TO_LAUNCHER_LIST_AICORE(PermuteCustom, OP_INPUT(input), OP_OUTPUT(output), OP_ATTR(perm));

        return output;
    }
} // namespace l0op