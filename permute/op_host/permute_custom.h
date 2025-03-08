#ifndef OP_API_INC_LEVEL0_OP_PERMUTE_CUSTOM_OP_H_
#define OP_API_INC_LEVEL0_OP_PERMUTE_CUSTOM_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
    aclTensor* PermuteCustom(const aclTensor* input, const aclIntArray *perm, aclOpExecutor* executor);
}

#endif // OP_API_INC_LEVEL0_OP_PERMUTE_CUSTOM_OP_H_