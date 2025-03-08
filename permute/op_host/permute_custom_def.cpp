#include "register/op_def_registry.h"

#include "permute_custom_tiling_data.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

#include <vector>
#include <iostream>

namespace optiling {

    static constexpr uint32_t BLOCK_SIZE = 32;
    static constexpr uint32_t BUFFER_NUM = 2;
    static constexpr uint32_t VECTOR_COMPUTE_UNIT_SIZE = 256;

    template<typename T>
    static std::vector<T> computeStrides(const std::vector<T>& shape) {
        auto dimNum = shape.size();
        std::vector<T> strides(dimNum);
        T product = 1;  // 累积乘积
        
        // 从最后一个维度向前计算步长
        auto i = dimNum;
        while (i-- > 0) {
            strides[i] = product;
            product *= shape[i];
        }

        return strides;
    }

    static std::vector<int64_t> opShapeToVector(const gert::Shape& shape) {
        std::vector<int64_t> vShape(shape.GetDimNum());
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            vShape[i] = shape[i];
        }

        return vShape;
    }

    template<typename T>
    static void printlnTuple(const std::vector<T>& tuple, const std::string& name = "") {
        auto size = tuple.size();
        auto lastIndex = size - 1;
        std::cout << name << "(";
        for(size_t i = 0; i < size; i++) {
            std::cout << tuple[i];
            if (i < lastIndex) {
                std::cout << ", ";
            }
        }
        std::cout << ")" << std::endl;
    }

    static ge::graphStatus PermuteCustomTilingFunc(gert::TilingContext* context)
    {
        const auto inputOpShape = context->GetInputShape(0)->GetStorageShape();
        const auto inputShape = opShapeToVector(inputOpShape);
        printlnTuple(inputShape, "inputShape");

        const auto attr0 = context->GetAttrs()->GetListInt(0);
        const auto permDim = attr0->GetSize();
        const auto permPtr = attr0->GetData();

        const std::vector<int64_t> perm(permPtr, permPtr + permDim);
        printlnTuple(perm, "perm");

        const auto inputShapeDim = inputShape.size();

        std::cout << "permDim = " << permDim << std::endl;

        std::cout << "inputShapeDim = " << inputShapeDim << std::endl;

        if (permDim != inputShapeDim) {
            return ge::GRAPH_PARAM_INVALID;
        }

        PermuteCustomTilingData *tilingData = context->GetTilingData<PermuteCustomTilingData>();
        tilingData->shapeDim = inputShapeDim;

        auto originShapeStrides = computeStrides(inputShape);
        printlnTuple(originShapeStrides, "originShapeStrides");

        for(size_t i = 0; i < inputShapeDim; i++) {
            auto dim = perm[i];
            tilingData->perm[i] = dim;
            tilingData->targetShape[i] = inputShape[dim];
            tilingData->originShapeStrides[i] = originShapeStrides[i];
        }

        auto platformInfo = context->GetPlatformInfo();
        
        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

        // 输入的元素数量
        const auto inputElementNum = inputOpShape.GetShapeSize();

        uint32_t elementSize = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), elementSize);
        

        std::cout << "elementSize = " << elementSize << std::endl;

        // 一个数据块中的元素数量
        const auto dataBlockElementNum = BLOCK_SIZE / elementSize;

        std::cout << "dataBlockElementNum = " << dataBlockElementNum << std::endl;

        // 向上对齐到BLOCK_SIZE(32字节)后的数据块数量
        const auto inputDataBlockNum = (inputElementNum + dataBlockElementNum - 1) / dataBlockElementNum;

        uint64_t ubMemSize;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubMemSize);

        // 可用的内存大小，输入占1份，输出占1份, 中间变量计算占4份, 外加输入张量的维度
        const auto memPartCount = 6 + inputShapeDim;
        const auto availableMemSize = ubMemSize / memPartCount;
        // 可用内存可容纳的数据块(32字节)数量
        const auto tileDataBlockNum = availableMemSize / BLOCK_SIZE;

        std::cout << "availableMemSize = " << availableMemSize << std::endl;
        std::cout << "tileDataBlockNum = " << tileDataBlockNum << std::endl;

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
        
        std::cout << "preCoreDataBlockNum = " << preCoreDataBlockNum << std::endl;
        std::cout << "inputTailDataBlockNum = " << inputTailDataBlockNum << std::endl;

        const auto smallChunkDataBlockNum  = preCoreDataBlockNum;
        const auto smallChunkTileNum = (smallChunkDataBlockNum + tileDataBlockNum -1) / tileDataBlockNum;
        auto smallChunkTailDataBlockNum = smallChunkDataBlockNum % tileDataBlockNum;
        if (smallChunkTailDataBlockNum == 0) {
            smallChunkTailDataBlockNum = smallChunkDataBlockNum;
        }

        std::cout << "smallChunkDataBlockNum = " << smallChunkDataBlockNum << std::endl;
        std::cout << "smallChunkTileNum = " << smallChunkTileNum << std::endl;
        std::cout << "smallChunkTailDataBlockNum = " << smallChunkTailDataBlockNum << std::endl;

        const auto bigChunkDataBlockNum  = preCoreDataBlockNum + 1;
        const auto bigChunkTileNum = (bigChunkDataBlockNum + tileDataBlockNum -1) / tileDataBlockNum;
        auto bigChunkTailDataBlockNum = bigChunkDataBlockNum % tileDataBlockNum;
        if (bigChunkTailDataBlockNum == 0) {
            bigChunkTailDataBlockNum = bigChunkDataBlockNum;
        }

        if (elementSize >= sizeof(float)) {
            tilingData->tileElementNum = tileDataBlockNum * dataBlockElementNum;
        }
        else {
            tilingData->tileElementNum = tileDataBlockNum * (BLOCK_SIZE / sizeof(float));
        }
        
        tilingData->inputTailDataBlockNum = inputTailDataBlockNum;
        tilingData->smallChunkElementNum = smallChunkDataBlockNum * dataBlockElementNum;
        tilingData->smallChunkTileNum = smallChunkTileNum;
        tilingData->smallChunkTailElementNum = smallChunkTailDataBlockNum * dataBlockElementNum;
        tilingData->bigChunkElementNum = bigChunkDataBlockNum * dataBlockElementNum;
        tilingData->bigChunkTileNum = bigChunkTileNum;
        tilingData->bigChunkTailElementNum = bigChunkTailDataBlockNum * dataBlockElementNum;

        std::cout << "ubMemSize = " << ubMemSize << std::endl;
        std::cout << "coreNum = " << coreNum << std::endl;
        std::cout << "inputTailDataBlockNum = " << tilingData->inputTailDataBlockNum << std::endl;
        std::cout << "tileElementNum = " << tilingData->tileElementNum << std::endl;
        std::cout << "smallChunkElementNum = " << tilingData->smallChunkElementNum << std::endl;
        std::cout << "smallChunkTileNum = " << tilingData->smallChunkTileNum << std::endl;
        std::cout << "smallChunkTailElementNum = " << tilingData->smallChunkTailElementNum << std::endl;
        std::cout << "bigChunkElementNum = " << tilingData->bigChunkElementNum << std::endl;
        std::cout << "bigChunkTileNum = " << tilingData->bigChunkTileNum << std::endl;
        std::cout << "bigChunkTailElementNum = " << tilingData->bigChunkTailElementNum << std::endl;
        

        context->SetBlockDim(coreNum);

        // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
        uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        std::cout << "sysWorkspaceSize = " << sysWorkspaceSize << std::endl;

        uint32_t userWorkspaceSize = tilingData->tileElementNum * tilingData->shapeDim * sizeof(float);
        std::cout << "userWorkspaceSize = " << userWorkspaceSize << std::endl;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context) {
        const gert::Shape* inputShape = context->GetInputShape(0);
        gert::Shape* outputShape = context->GetOutputShape(0);

        const auto attr0 = context->GetAttrs()->GetListInt(0);
        const auto perm = attr0->GetData();

        for(size_t i = 0; i < attr0->GetSize(); i++) {
            outputShape->SetDim(i, inputShape->GetDim(perm[i]));
        }
        
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops {
    class PermuteCustom : public OpDef {
    public:
        explicit PermuteCustom(const char* name) : OpDef(name)
        {
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT32, ge::DT_BOOL})
                .FormatList({ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT32, ge::DT_BOOL})
                .FormatList({ge::FORMAT_ND});
            this->Attr("perm").ListInt();
    
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::PermuteCustomTilingFunc)
                .AddConfig("ascend310b");
        }
    };
    
    OP_ADD(PermuteCustom);
}


    