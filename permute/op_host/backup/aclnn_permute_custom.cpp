#include <string.h>
#include "graph/types.h"
#include "aclnn_permute_custom.h"
#include "permute_custom.h"

#include "aclnn_kernels/contiguous.h"

// 用于定义单算子API执行接口依赖的AscendCL meta接口，通过这些元接口可构建不同数据结构，如aclTensor、aclScalar、aclIntArray等。
#include "aclnn/acl_meta.h" // libnnopbase.so
// 用于定义单算子API执行接口的公共base接口，即aclnnInit和aclnnFinalize。
#include "aclnn/aclnn_base.h" // libnnopbase.so

#include "opdev/make_op_executor.h"

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnPermuteCustomGetWorkspaceSize(
    const aclTensor *input,
    const aclIntArray *perm,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnPermuteCustom, DFX_IN(input, perm), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    aclOpExecutor *l0Executor = uniqueExecutor.get();

    auto l0PermuteOut = l0op::PermuteCustom(input, perm, l0Executor);

    auto viewCpoyRet = l0op::ViewCopy(l0PermuteOut, out, l0Executor);

    if (viewCpoyRet == nullptr) {
        std::cout << "viewCpoyRet is null" << std::endl;
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPermuteCustom(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnPermuteCustom);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
