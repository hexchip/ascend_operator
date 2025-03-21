
#include "cumsum_custom_tiling_data.h"
#include "register/op_def_registry.h"

#include <cmath>
#include <vector>
#include <utils/type_utils.h>


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

    double logbase(double a, double base) {
        return log(a) / log(base);
    }

    static std::vector<uint32_t> compute_hierarchy_levels(uint32_t dataSize, uint32_t blockSize) {
        std::vector<uint32_t> hierarchy;

        auto blockNum = (dataSize + blockSize - 1) / blockSize;
        hierarchy.push_back(blockNum);
        while (blockNum > blockSize) {
            blockNum = (blockNum + blockSize - 1) / blockSize;
            hierarchy.push_back(blockNum);
        }

        return hierarchy;
    }

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        const auto inputOpShape = context->GetInputShape(0)->GetStorageShape();
        const auto inputShape = opShapeToVector(inputOpShape);
        printlnTuple(inputShape, "inputShape");

        const auto inputShapeDim = inputShape.size();

        if (inputShapeDim != 2) {
            return ge::GRAPH_PARAM_INVALID;
        }

        const auto row = inputShape[0];
        const auto col = inputShape[1];

        uint32_t elementSize = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), elementSize);
        std::cout << "elementSize = " << elementSize << std::endl;

        const auto colBlockElementNum = VECTOR_COMPUTE_UNIT_SIZE / elementSize;
        const auto colBlockNum = (col + colBlockElementNum - 1) / colBlockElementNum;

        CumsumCustomTilingData *tilingData = context->GetTilingData<CumsumCustomTilingData>();
        tilingData->shapeDim = inputShapeDim;

        auto platformInfo = context->GetPlatformInfo();
        
        const auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);

        uint64_t ubMemSize = 262144;
        // uint64_t ubMemSize;
        // ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubMemSize);

        // 可用的内存大小，输入占1份，输出占1份, 中间变量计算占4份, 外加输入张量的维度
        const auto memPartCount = 6 + inputShapeDim;
        const auto availableMemSize = ubMemSize / memPartCount;
        // 可用内存可容纳的数据块(BLOCK_SIZE字节)数量
        const auto availableDataBlockNum = availableMemSize / BLOCK_SIZE;

        std::cout << "availableMemSize = " << availableMemSize << std::endl;
        std::cout << "tileDataBlockNum = " << tileDataBlockNum << std::endl;

        // 一个数据块中的元素数量
        const auto dataBlockElementNum = BLOCK_SIZE / elementSize;
        const auto requiredDataBlockNum = (col + dataBlockElementNum - 1) / dataBlockElementNum;
        const auto tileNum = (requiredDataBlockNum + availableDataBlockNum - 1) / availableDataBlockNum;
        const auto tileElementNum = availableDataBlockNum * dataBlockElementNum;
        const auto colTailElementNum = col % tileElementNum;



        context->SetBlockDim(row);

        // 如需要使用系统workspace需要调用GetLibApiWorkSpaceSize获取系统workspace的大小。
        // uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        // std::cout << "sysWorkspaceSize = " << sysWorkspaceSize << std::endl;

        // uint32_t userWorkspaceSize = tilingData->tileElementNum * tilingData->shapeDim * sizeof(float);
        // std::cout << "userWorkspaceSize = " << userWorkspaceSize << std::endl;
        size_t *currentWorkspace = context->GetWorkspaceSizes(1); // 通过框架获取workspace的指针，GetWorkspaceSizes入参为所需workspace的块数。当前限制使用一块。
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}


namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* x1_shape = context->GetInputShape(0);
        gert::Shape* y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
    }
}


namespace ops {
    class CumsumCustom : public OpDef {
    public:
        explicit CumsumCustom(const char* name) : OpDef(name)
        {
            this->Input("input")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT8, ge::DT_INT16, ge::DT_INT32, ge::DT_UINT8, ge::DT_UINT16, ge::DT_UINT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("dim").Int();

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310b");

        }
    };

    OP_ADD(CumsumCustom);
}
