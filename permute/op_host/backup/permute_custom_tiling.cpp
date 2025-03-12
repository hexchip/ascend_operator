#include <vector>

#include "fmt/ranges.h"

#include "permute_custom_tiling.h"
#include "permute_custom_tiling_data.h"

#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "toolchain/slog.h"
#include "aclnn/opdev/shape_utils.h"

namespace optiling {

    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t VECTOR_COMPUTE_UNIT_SIZE = 256;

    std::string format_as(PermuteCustomTilingData data) {
        
        auto originShape = std::vector(std::begin(data.originShape), std::end(data.originShape));
        auto perm = std::vector(std::begin(data.perm), std::end(data.perm));
        auto targetShape = std::vector(std::begin(data.targetShape), std::end(data.targetShape));

        return fmt::format("PermuteCustomTilingData["
                "\t shapeDim={}\n"
                "\t originShape={}\n"
                "\t perm={}\n"
                "\t targetShape={}\n"
                "\t tileElementNum={} \n"
                "\t inputTailDataBlockNum={}\n"
                "\t smallChunkElementNum={}\n"
                "\t smallChunkTileNum={}\n"
                "\t smallChunkTailElementNum={}\n"
                "\t bigChunkElementNum={}\n"
                "\t bigChunkTileNum={}\n"
                "\t bigChunkTailElementNum={}\n"
            "]", 
            data.shapeDim, 
            originShape, 
            perm, 
            targetShape, 
            data.tileElementNum,
            data.inputTailDataBlockNum,
            data.smallChunkElementNum,
            data.smallChunkTileNum,
            data.smallChunkTailElementNum,
            data.bigChunkElementNum,
            data.bigChunkTileNum,
            data.bigChunkTailElementNum
        ); 
    }

    ge::graphStatus PermuteCustomTilingFunc(gert::TilingContext* context)
    {
        const auto inputShape = context->GetInputShape(0)->GetStorageShape();

        const auto attr0 = context->GetAttrs()->GetListInt(0);
        const auto permDim = attr0->GetSize();
        const auto permPtr = attr0->GetData();

        const std::vector<int64_t> perm(permPtr, permPtr + permDim);

        const auto inputShapeDim = inputShape.GetDimNum();

        if (permDim != inputShapeDim) {
            DlogSub(OP, "Tiling", DLOG_ERROR, "Invalid permutation %s for shape %s", fmt::format("{}", perm), op::ToString(inputShape).GetString());

            fmt::println("Invalid permutation {} for shape {}", perm, op::ToString(inputShape).GetString());
            return ge::GRAPH_PARAM_INVALID;
        }

        PermuteCustomTilingData *tilingData = context->GetTilingData<PermuteCustomTilingData>();

        tilingData->shapeDim = inputShapeDim;
        for(size_t i = 0; i < inputShapeDim; i++) {
            tilingData->originShape[i] = inputShape[i];
            tilingData->perm[i] = perm[i];
            tilingData->targetShape[i] = inputShape[perm[i]];
        }

        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

        // 输入的元素数量
        const auto inputElementNum = inputShape.GetShapeSize();

        uint32_t elementSize = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), elementSize);
        
        // 一个数据块中的元素数量
        const auto dataBlockElementNum = BLOCK_SIZE / elementSize;

        // 向上对齐到BLOCK_SIZE(32字节)后的数据块数量
        const auto inputDataBlockNum = (inputElementNum + dataBlockElementNum - 1) / dataBlockElementNum;

        uint64_t ubMemSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubMemSize);

        // 可用的内存大小，输入和输出各占一份内存, 中间变量计算占一份内存
        const auto availableMemSize = ubMemSize / 3;
        // 可用内存可容纳的数据块(32字节)数量
        const auto tileDataBlockNum = availableMemSize / BLOCK_SIZE;

        // 向量计算单元一次计算可处理的数据块数量
        constexpr auto vectorComputeUnitBlockNum = VECTOR_COMPUTE_UNIT_SIZE / BLOCK_SIZE;
        // 需要使用向量计算单元进行计算的次数
        const auto processCount = (inputDataBlockNum + vectorComputeUnitBlockNum -1) / vectorComputeUnitBlockNum;
        // 设备拥有的向量计算单元数量
        auto coreNum = ascendcPlatform.GetCoreNum();
        // 最终使用的向量计算单元数量
        coreNum = (processCount > coreNum) ? coreNum : processCount;

        // 每个核需要处理的数据块数量
        const auto preCoreDataBlockNum = inputDataBlockNum / coreNum;
        // 剩余的数据块分配给部分核，会有部分核多算一个数据块
        const auto inputTailDataBlockNum = inputDataBlockNum % coreNum;
        
        const auto smallChunkDataBlockNum  = preCoreDataBlockNum;
        const auto smallChunkTileNum = (smallChunkDataBlockNum + tileDataBlockNum -1) / tileDataBlockNum;
        const auto smallChunkTailDataBlockNum = smallChunkDataBlockNum % tileDataBlockNum;

        const auto bigChunkDataBlockNum  = preCoreDataBlockNum + 1;
        const auto bigChunkTileNum = (bigChunkDataBlockNum + tileDataBlockNum -1) / tileDataBlockNum;
        const auto bigChunkTailDataBlockNum = bigChunkDataBlockNum % tileDataBlockNum;

        tilingData->tileElementNum = tileDataBlockNum * dataBlockElementNum;
        tilingData->inputTailDataBlockNum = inputTailDataBlockNum;
        tilingData->smallChunkElementNum = smallChunkDataBlockNum * dataBlockElementNum;
        tilingData->smallChunkTileNum = smallChunkTileNum;
        tilingData->smallChunkTailElementNum = smallChunkTailDataBlockNum * dataBlockElementNum;
        tilingData->bigChunkElementNum = bigChunkDataBlockNum * dataBlockElementNum;
        tilingData->bigChunkTileNum = bigChunkTileNum;
        tilingData->bigChunkTailElementNum = bigChunkTailDataBlockNum * dataBlockElementNum;

        fmt::println("{}", *tilingData);

        return ge::GRAPH_SUCCESS;
    }

    IMPL_OP(PermuteCustom).Tiling(PermuteCustomTilingFunc);
}