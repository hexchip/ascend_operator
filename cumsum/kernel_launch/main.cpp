#include <iostream>
#include <cmath>
#include <vector>
#include <exception>
#include <algorithm> // for std::copy
#include <numeric> // for std::accumulate
#include <bitset>
#include <cstdint>

#include "boost/dynamic_bitset.hpp"

#include "cumsum_custom_tiling_data.h"
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
// #include "aclrtlaunch_cumsum_custom.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void cumsum_custom(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, optiling::CumsumCustomTilingData tilingData);
#endif

static constexpr uint32_t BLOCK_SIZE = 32;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t VECTOR_COMPUTE_UNIT_SIZE = 256;

static std::vector<uint32_t> computeCumsumTileHierarchy(uint32_t dataSize, uint32_t blockSize) {
    std::vector<uint32_t> hierarchy;

    auto blockNum = (dataSize + blockSize - 1) / blockSize;
    hierarchy.push_back(blockNum);
    while (blockNum > blockSize) {
        blockNum = (blockNum + blockSize - 1) / blockSize;
        hierarchy.push_back(blockNum);
    }

    hierarchy.push_back(1);

    return hierarchy;
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


static uint64_t computeShapeSize(std::vector<uint32_t> inputShape) {
    uint64_t size = 1;
    for (auto dim : inputShape) {
        size *= dim;
    }

    return size;
}

template<typename T>
static std::vector<T> convertBitset(const boost::dynamic_bitset<>& mask) {
    static_assert(std::is_unsigned_v<T>, "T must be unsigned integral type");
    constexpr size_t typeBitNum = sizeof(T) * 8;
    static_assert(typeBitNum <= 64 && (typeBitNum & (typeBitNum - 1)) == 0, "typeBitNum must be power-of-two (8/16/32/64 bits)");

    const auto maskSize = mask.size();
    const size_t partNum = (maskSize + typeBitNum - 1) / typeBitNum;

    std::vector<T> parts(partNum);

    const size_t frontPartNum = maskSize / typeBitNum;
    for (size_t i = 0; i < frontPartNum; ++i) {
        const size_t shift = i * typeBitNum;

        T value = 0;
        for (size_t j = 0; j < typeBitNum; ++j) {
            const size_t offset = shift + j;
            if (mask[offset]) {
                value |= T{1} << j;
            }
        }

        parts[i] = value;
    }


    const size_t remainingBitNum = maskSize % typeBitNum;
    const size_t lastIndex = partNum - 1;
    if (remainingBitNum != 0) {
        const size_t offset = frontPartNum * typeBitNum;
        T value = 0;
        
        for (size_t i = 0; i < remainingBitNum; ++i) {
            if (mask[offset + i]) {
                value |= T{1} << i;
            }
        }
        parts[lastIndex] = value;
    }

    return parts;
}

template<typename T>
static std::vector<std::vector<T>> generateHillisSteeleScanGatherMaskPatterns(uint32_t dataSize) {
    const uint32_t zeroPaddingCount = dataSize / 2;
    boost::dynamic_bitset<> mask0(zeroPaddingCount);
    std::string mask0String;
    boost::to_string(mask0, mask0String);
    
    boost::dynamic_bitset<> mask1(dataSize);
    std::string mask1String;
    boost::to_string(mask1.flip(), mask1String);
    
    boost::dynamic_bitset<> padedMask(mask1String + mask0String);

    std::vector<std::vector<T>> result;

    uint32_t stride = 1;
    while(stride < dataSize) {
        const auto maks = padedMask >> stride;
        std::cout << maks << std::endl;
        result.emplace_back(convertBitset<uint32_t>(maks));
        stride *= 2;
    }

    return result;
}

static optiling::CumsumCustomTilingData TilingFunc(std::vector<uint32_t> inputShape, uint32_t elementSize) {
    printlnTuple(inputShape, "inputShape");

    const auto inputShapeDim = inputShape.size();

    if (inputShapeDim != 2) {
        throw std::invalid_argument("inputShapeDim must is 2");
    }

    const auto col = inputShape[1];

    // 一个数据块中的元素数量
    const auto dataBlockElementNum = BLOCK_SIZE / elementSize;
    // 每一行的数据块数量
    const auto rowDataBlockNum = (col + dataBlockElementNum - 1) / dataBlockElementNum;
    // 经过对齐后，每一行的元素数量
    const auto alignedRowElementNum = rowDataBlockNum * dataBlockElementNum;
    // 对齐会产生冗余数据，冗余数据不能参与计算
    const auto redundantElementNum = alignedRowElementNum - col;

    uint64_t ubMemSize = 262144;
    // 输入占1份，输出占1份, 中间变量计算占1.1份
    const auto memPartCount = 3.1f;
    // 可用的内存大小
    const auto availableMemSize = (uint64_t)(ubMemSize / memPartCount);
    std::cout << "availableMemSize = " << availableMemSize << std::endl;

    // 可用内存可容纳的数据块(${BLOCK_SIZE}字节)数量
    const auto availableDataBlockNum = availableMemSize / BLOCK_SIZE;
    // 经过对齐后，可用内存可容纳的元素数量, 即分块的元素数量
    const auto tileElementNum = availableDataBlockNum * dataBlockElementNum;
    // 一行数据需要分几块
    const auto tileNum = (alignedRowElementNum + tileElementNum - 1) / tileElementNum;
    // 分块有可能不被数据填满，计算尾部数据的数量
    const auto tailDataElementNum = alignedRowElementNum % tileElementNum;
    // 如果分块刚好被填满了，此时余数为0，需要特殊处理。
    // 尾部数据还需要去除冗余数据
    const auto finalTailDataElementNum = (tailDataElementNum == 0 ? tileElementNum : tailDataElementNum) - redundantElementNum;
    // 尾部数据还需要向${BLOCK_SIZE}字节对齐
    const auto alignedFinalTailDataElementNum = (finalTailDataElementNum + dataBlockElementNum - 1) / dataBlockElementNum * dataBlockElementNum;

    optiling::CumsumCustomTilingData tilingData;

    tilingData.col = col;
    tilingData.alignedRowElementNum = alignedRowElementNum;
    tilingData.tileElementNum = tileElementNum;
    tilingData.tileNum = tileNum;
    tilingData.tailDataElementNum = finalTailDataElementNum;
    tilingData.alignedTailDataElementNum = alignedFinalTailDataElementNum;

    const auto cumsumTileElementNum = VECTOR_COMPUTE_UNIT_SIZE / elementSize; 
    tilingData.cumsumTileElementNum = cumsumTileElementNum;
    
    // 分块数据的cumsum分块层级
    const auto cumsumTileHierarchy = computeCumsumTileHierarchy(tileElementNum, cumsumTileElementNum);
    std::copy(cumsumTileHierarchy.begin(), cumsumTileHierarchy.end(), tilingData.cumsumTileHierarchy);
    tilingData.cumsumTotalTileElementNum = std::accumulate(cumsumTileHierarchy.begin(), cumsumTileHierarchy.end(), 0);

    // 尾部数据的cumsum分块层级
    const auto tailDataCumsumTileHierarchy = computeCumsumTileHierarchy(finalTailDataElementNum, cumsumTileElementNum);
    std::copy(tailDataCumsumTileHierarchy.begin(), tailDataCumsumTileHierarchy.end(), tilingData.tailDataCumsumTileHierarchy);

    const auto maskPatterns = generateHillisSteeleScanGatherMaskPatterns<uint32_t>(cumsumTileElementNum);
    for (size_t i = 0; i < maskPatterns.size(); ++i) {
        for (size_t j = 0; j < maskPatterns[i].size(); ++j) {
            tilingData.hillisSteeleScanGatherMaskPatterns[i][j] = maskPatterns[i][j];
        }
    }

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%x\n", tilingData.hillisSteeleScanGatherMaskPatterns[i][j]);
        }

        printf("\n");
    }

    return tilingData;
}

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 2;

    std::vector<uint32_t> inputShape = {2, 6};

    auto elementSize = sizeof(uint16_t);

    const auto tilingData = TilingFunc(inputShape, elementSize);

    return 0;

    size_t inputByteSize = computeShapeSize(inputShape) * elementSize;
    size_t outputByteSize = inputByteSize;
    size_t inputBufferByteSize = inputShape[0] * tilingData.alignedRowElementNum * elementSize;
    size_t outputBufferByteSize = inputBufferByteSize;

#ifdef ASCENDC_CPU_DEBUG
    void *input = AscendC::GmAlloc(inputBufferByteSize);
    void *output = AscendC::GmAlloc(outputBufferByteSize);

    ReadFile("./input/input.bin", inputByteSize, input, inputBufferByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(cumsum_custom, blockDim, (uint8_t*)input, (uint8_t*)output, (uint8_t*)0, tilingData); // use this macro for cpu debug

    WriteFile("./output/output.bin", output, outputByteSize);

    AscendC::GmFree(input);
    AscendC::GmFree(output);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    AddCustomTilingData *tiling;
    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize));
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/input_y.bin", inputByteSize, yHost, inputByteSize);

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputByteSize, yHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(add_custom)(blockDim, stream, xDevice, yDevice, zDevice, tiling);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(tiling));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}