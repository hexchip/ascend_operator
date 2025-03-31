#include "kernel_operator.h"
#include "cumsum_custom_tiling_data.h"

static constexpr uint32_t DATA_BLOCK_SIZE = 32;
static constexpr size_t BUFFER_NUM = 1;

template<typename T>
class KernelCumsum {
public:
    __aicore__ inline KernelCumsum(optiling::CumsumCustomTilingData& tilingData) : tilingData(tilingData) {

    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace)
    {
        auto currentRow = AscendC::GetBlockIdx();

        auto globalBufferOffset = currentRow * tilingData.col;

        inGm.SetGlobalBuffer((__gm__ T *)input + globalBufferOffset, tilingData.alignedRowElementNum);

        outGm.SetGlobalBuffer((__gm__ T *)output + globalBufferOffset, tilingData.alignedRowElementNum);
    }

    __aicore__ inline void Process(AscendC::TPipe& pipe)
    {
       
        uint32_t bufferSize = tilingData.tileElementNum * sizeof(T);
        pipe.InitBuffer(calcBuffer, bufferSize);
        pipe.InitBuffer(blockSumBuffer, tilingData.cumsumTotalTileElementNum);
        pipe.InitBuffer(inQueue, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, bufferSize);

        processElementNum = tilingData.tileElementNum;
        alignedProcessElementNum = processElementNum;
        cumsumTileHierarchy = tilingData.cumsumTileHierarchy;
        uint32_t loopCount = tilingData.tileNum;
        for (uint32_t i = 0; i < loopCount; i++) {
            if (i == loopCount - 1) {
                processElementNum = tilingData.tailDataElementNum;
                alignedProcessElementNum = tilingData.alignedTailDataElementNum;
                cumsumTileHierarchy = tilingData.tailDataCumsumTileHierarchy;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
        AscendC::DataCopy(inLocal, inGm[progress * tilingData.tileElementNum], alignedProcessElementNum);

        inQueue.EnQue<T>(inLocal);
    }

    __aicore__ inline void HillisSteeleScan(const AscendC::LocalTensor<T>& data, uint32_t dataSize) {
        const auto cumsumTileElementNum = tilingData.cumsumTileElementNum;

        uint32_t bufSize = cumsumTileElementNum;
        const auto shifted = calcBuffer.Get<float>(bufSize);
        uint32_t bufOffset = bufSize * sizeof(float);

        uint32_t zeroPaddingCount = cumsumTileElementNum / 2;
        bufSize = zeroPaddingCount + cumsumTileElementNum;
        const auto gatherMaskSrc = calcBuffer.GetWithOffset<float>(bufSize, bufOffset);
        bufOffset += bufSize * sizeof(float);

        AscendC::Duplicate<float>(gatherMaskSrc, 0, bufSize);

        uint32_t dataStartOffset = zeroPaddingCount;
        AscendC::Cast(gatherMaskSrc[dataStartOffset], data, AscendC::RoundMode::CAST_NONE, cumsumTileElementNum);

        bufOffset = (bufOffset + 32 - 1) / 32;
        bufSize = gatherMaskSrc.GetSize() / 32;
        const auto gatherMaskPattern = calcBuffer.GetWithOffset<uint32_t>(bufSize, bufOffset);
        AscendC::Duplicate<uint32_t>(gatherMaskPattern, 0, bufSize);

        int16_t stride = 1;
        uint32_t loop = 0;
        while(stride < dataSize) {
            for(uint32_t i = 0; i < gatherMaskPattern.GetSize(); i++) {
                gatherMaskPattern.SetValue(i, tilingData.hillisSteeleScanGatherMaskPatterns[loop][i])
            }

            uint32_t mask = gatherMaskSrc.GetSize();
            uint64_t rsvdCnt = 0;
            AscendC::GatherMask(gatherMaskSrc, gatherMaskSrc, gatherMaskPattern, true, mask, {1, 2, 8, 1}, rsvdCnt);

            AscendC::Duplicate<T>(shifted, 0, stride);
            AscendC::Add(data, data, shifted, dataSize);
            stride *= 2;
            loop++;
        }
    }

    __aicore__ inline void CumSum(
        const AscendC::LocalTensor<T>& data,
        uint32_t dataSize,
        uint32_t blockSize, 
        const AscendC::LocalTensor<T>& blockSums,
        uint32_t blockSumsListOffset = 0,
        uint32_t level = 0) {

        const auto blockNum = cumsumTileHierarchy[level];

        for(uint32_t i = 0; i < blockNum; ++i) {
            const auto startOffset = i * blockSize;
            const auto endOffset = startOffset + blockSize;
            const auto finalBlockSize = (endOffset > dataSize ? dataSize : endOffset) - startOffset;
            auto block = data[startOffset];
            HillisSteeleScan(block, finalBlockSize);
            blockSums.SetValue(i, block.GetValue(finalBlockSize - 1));
        }

        if (blockNum > 1) {
            const auto nextBlockSumsListOffset = blockSumsListOffset + blockNum;
            const auto nextBlockSums = blockSums[nextBlockSumsListOffset];
            CumSum(blockSums, blockNum, blockSize, nextBlockSums, nextBlockSumsListOffset, level + 1);

            const auto broadcastedAdjustment = calcBuffer.Get<T>(blockSize);
            // Step 3: 向量化叠加修正值（排除第一个块）
            for(uint32_t i = 1; i < blockNum; ++i) {
                const auto startOffset = i * blockSize;
                const auto endOffset = startOffset + blockSize;
                const auto finalBlockSize = (endOffset > dataSize ? dataSize : endOffset) - startOffset;
                const auto block = data[startOffset];
                // 生成修正值向量（广播adjustments[i-1]到整个向量）
                AscendC::Duplicate<T>(broadcastedAdjustment, blockSums(i-1), finalBlockSize);
                AscendC::Add(block, block, broadcastedAdjustment, finalBlockSize);
            }
        }
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<T> inLocal = inQueue.DeQue<T>();

        const auto blockSumsList = blockSumBuffer.Get<T>();
        CumSum(inLocal, processElementNum, tilingData.cumsumTileElementNum, blockSumsList);

        AscendC::LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        AscendC::DataCopy(outLocal, inLocal, alignedProcessElementNum);
        outQueue.EnQue<T>(outLocal);

        inQueue.FreeTensor(inLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<T> outLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(outGm[progress * tilingData.tileElementNum], outLocal, alignedProcessElementNum);
        outQueue.FreeTensor(outLocal);
    }

private:
    const optiling::CumsumCustomTilingData& tilingData;

    static constexpr auto elementSize = sizeof(T);

    uint32_t processElementNum;
    uint32_t alignedProcessElementNum;
    const uint32_t *cumsumTileHierarchy;

    AscendC::TBuf<> calcBuffer;
    AscendC::TBuf<> blockSumBuffer;

    AscendC::GlobalTensor<T> inGm;
    AscendC::GlobalTensor<T> outGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
};

// extern "C" __global__ __aicore__ void cumsum_custom(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
//     REGISTER_TILING_DEFAULT(optiling::CumsumCustomTilingData);
//     GET_TILING_DATA(tilingData, tiling);
//     KernelCumsum<DTYPE_INPUT> op(tilingData);
//     op.Init(input, output, workspace);
//     AscendC::TPipe pipe;
//     op.Process(pipe);
// }

extern "C" __global__ __aicore__ void cumsum_custom(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, optiling::CumsumCustomTilingData tilingData) {
    KernelCumsum<half> op(tilingData);
    op.Init(input, output, workspace);
    AscendC::TPipe pipe;
    op.Process(pipe);
}