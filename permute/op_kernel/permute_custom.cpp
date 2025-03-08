#include "kernel_operator.h"
#include "permute_custom_tiling_data.h"

// tensor num for each queue
constexpr uint32_t BUFFER_NUM = 1;

template<typename T>
class KernelPermute {
public:
    __aicore__ inline KernelPermute(optiling::PermuteCustomTilingData& tilingData) : tilingData(tilingData) {

    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, GM_ADDR workspace)
    {
        auto coreIdx = AscendC::GetBlockIdx();

        long chunkElementNum;
        long globalBufferOffset;

        if (coreIdx < tilingData.inputTailDataBlockNum) {
            chunkElementNum = tilingData.bigChunkElementNum;
            this->chunkTailElementNum = tilingData.bigChunkTailElementNum;
            this->chunkTileNum = tilingData.bigChunkTileNum;
            globalBufferOffset = chunkElementNum * coreIdx;
        }
        else {
            chunkElementNum = tilingData.smallChunkElementNum;
            this->chunkTailElementNum = tilingData.smallChunkTailElementNum;
            this->chunkTileNum = tilingData.smallChunkTileNum;
            globalBufferOffset = chunkElementNum * coreIdx + (tilingData.bigChunkElementNum - chunkElementNum) * tilingData.inputTailDataBlockNum;
        }

        this->chunkElementNum = chunkElementNum;

        inGm.SetGlobalBuffer((__gm__ T *)input + globalBufferOffset, chunkElementNum);
        outGm.SetGlobalBuffer((__gm__ T *)output + globalBufferOffset, chunkElementNum);
        // workspaceGm.SetGlobalBuffer((__gm__ float *)workspace, chunkElementNum * tilingData.shapeDim);

        this->tileElementNum = tilingData.tileElementNum;
        this->bufferSize = this->tileElementNum * sizeof(T);
        this->preCalcBufferSize = this->tileElementNum * sizeof(float);
    }

    __aicore__ inline void Process(AscendC::TPipe& pipe)
    {
        uint32_t loopCount = this->chunkTileNum;
        this->processElementNum = this->tileElementNum;

        // pipe.InitBuffer(workspaceOutQueue, 1, this->preCalcBufferSize);
        pipe.InitBuffer(calcBuffer, this->preCalcBufferSize * 4);
        // pipe.Reset();
        // pipe.InitBuffer(workspaceInQueue, BUFFER_NUM * 2, this->preCalcBufferSize);
        pipe.InitBuffer(indicesBuffer, this->preCalcBufferSize * this->tilingData.shapeDim);
        pipe.InitBuffer(inQueue, BUFFER_NUM, this->bufferSize);
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->bufferSize);

        for (uint32_t i = 0; i < loopCount; i++) {
            if (i == this->chunkTileNum - 1) {
                this->processElementNum = this->chunkTailElementNum;
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
        AscendC::DataCopy(inLocal, inGm[progress * this->tileElementNum], this->processElementNum);

        inQueue.EnQue<T>(inLocal);
    }

    __aicore__ inline void ReshapeTargetLinearIndex(uint32_t progress)
    {

        auto offset = progress * this->tileElementNum;

        AscendC::LocalTensor<float> remainderLocal = calcBuffer.Get<float>(this->processElementNum);

        AscendC::LocalTensor<int32_t> tmpLocal = calcBuffer.GetWithOffset<int32_t>(this->processElementNum, processElementNum * sizeof(int32_t));

        // errorStr: The instruction is not defined in ISA.
        // AscendC::CreateVecIndex(remainderLocal, (float)0, this->processElementNum);

        for(int32_t i = 0; i < this->processElementNum; ++i) {
            tmpLocal.SetValue(i, i + offset);
        }

        AscendC::Cast(remainderLocal, tmpLocal, AscendC::RoundMode::CAST_NONE, this->processElementNum);

        if (tilingData.shapeDim > 1) {
            
            AscendC::LocalTensor<float> shapeDimLocal = calcBuffer.GetWithOffset<float>(this->processElementNum, processElementNum * sizeof(float) * 2);
            AscendC::LocalTensor<float> quotientLocal = calcBuffer.GetWithOffset<float>(this->processElementNum, processElementNum * sizeof(float) * 3);

            for(auto i = tilingData.shapeDim - 1; i > 0; --i) {

                const auto dim = tilingData.targetShape[i];
             
                AscendC::Duplicate<int32_t>(tmpLocal, (int32_t)dim, this->processElementNum);
                AscendC::Cast(shapeDimLocal, tmpLocal, AscendC::RoundMode::CAST_TRUNC, this->processElementNum);

                AscendC::Div(quotientLocal, remainderLocal, shapeDimLocal, this->processElementNum);
                AscendC::Cast(tmpLocal, quotientLocal, AscendC::RoundMode::CAST_TRUNC, this->processElementNum);
                AscendC::Cast(quotientLocal, tmpLocal, AscendC::RoundMode::CAST_TRUNC, this->processElementNum);
           
                AscendC::LocalTensor<float> indexLocal = indicesBuffer.GetWithOffset<float>(this->processElementNum, this->processElementNum * sizeof(float) * i);
                AscendC::Mul(indexLocal, quotientLocal, shapeDimLocal, this->processElementNum);

                AscendC::Sub(indexLocal, remainderLocal, indexLocal, this->processElementNum);

                AscendC::DataCopy(remainderLocal, quotientLocal, this->processElementNum);
            }
        }

        AscendC::DataCopy(indicesBuffer.Get<float>(this->processElementNum), remainderLocal, this->processElementNum);

        AscendC::DumpTensor(indicesBuffer.Get<float>(), 123, this->processElementNum * tilingData.shapeDim);
    }

    __aicore__ inline AscendC::LocalTensor<float> ComputeTargetLinearIndex(uint32_t progress) {
        const auto offset = progress * this->tileElementNum;
        const auto processElementNum = this->processElementNum;

        const auto lastDimIndex = tilingData.perm[tilingData.shapeDim - 1];
        AscendC::LocalTensor<float> targetIndexLocal = indicesBuffer.GetWithOffset<float>(processElementNum, processElementNum * sizeof(float) * lastDimIndex);

        AscendC::LocalTensor<float> strideLocal = calcBuffer.GetWithOffset<float>(processElementNum, processElementNum * sizeof(float));
        AscendC::LocalTensor<int32_t> tmpLocal = calcBuffer.GetWithOffset<int32_t>(processElementNum, processElementNum * sizeof(int32_t) * 2);

        for(size_t i = 0; i < tilingData.shapeDim - 1; ++i) {
            auto stride = tilingData.originShapeStrides[i];
            AscendC::Duplicate<int32_t>(tmpLocal, (int32_t)stride, processElementNum);
            AscendC::Cast(strideLocal, tmpLocal, AscendC::RoundMode::CAST_TRUNC, processElementNum);
            AscendC::DumpTensor(strideLocal, 1, processElementNum);

            auto indexLocal = indicesBuffer.GetWithOffset<float>(processElementNum, processElementNum * sizeof(float) * tilingData.perm[i]);

            AscendC::DumpTensor(indexLocal, 2, processElementNum);

            AscendC::MulAddDst(targetIndexLocal, indexLocal, strideLocal, processElementNum);

            AscendC::DumpTensor(targetIndexLocal, 3, processElementNum);
        }

        return targetIndexLocal;
    }

    __aicore__ inline void Compute(uint32_t progress)
    {
        ReshapeTargetLinearIndex(progress);
        auto targetIndexLocal = ComputeTargetLinearIndex(progress);
        // const auto lastDimIndex = tilingData.perm[tilingData.shapeDim - 1];
        // AscendC::LocalTensor<float> targetIndexLocal = indicesBuffer.GetWithOffset<float>(processElementNum, processElementNum * sizeof(float) * lastDimIndex);
        AscendC::LocalTensor<float> elementSizeLocal = calcBuffer.GetWithOffset<float>(this->processElementNum, this->processElementNum * sizeof(int32_t));
        AscendC::LocalTensor<int32_t> tmpLocal = calcBuffer.GetWithOffset<int32_t>(this->processElementNum, this->processElementNum * sizeof(int32_t) * 2);

        AscendC::Duplicate<int32_t>(tmpLocal, sizeof(T), this->processElementNum);
        // To float
        AscendC::Cast(elementSizeLocal, tmpLocal, AscendC::RoundMode::CAST_TRUNC, this->processElementNum);
        AscendC::Mul(targetIndexLocal, targetIndexLocal, elementSizeLocal, this->processElementNum);
        // To int32_t
        AscendC::Cast(tmpLocal, targetIndexLocal, AscendC::RoundMode::CAST_TRUNC, this->processElementNum);
        // To uint32_t
        auto offsetLocal = tmpLocal.ReinterpretCast<uint32_t>();
        AscendC::DumpTensor(offsetLocal, 668, this->processElementNum);

        AscendC::LocalTensor<T> inLocal = inQueue.DeQue<T>();
        AscendC::LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
        AscendC::Gather(outLocal, inLocal, offsetLocal, 0, this->processElementNum);

        // AscendC::DataCopy(outLocal, inLocal, this->processElementNum);

        outQueue.EnQue<T>(outLocal);
        // workspaceInQueue.FreeTensor(targetIndexLocal);
        inQueue.FreeTensor(inLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<T> outLocal = outQueue.DeQue<T>();
        AscendC::DataCopy(outGm[progress * this->tileElementNum], outLocal, this->processElementNum);
        outQueue.FreeTensor(outLocal);
    }

private:
    const optiling::PermuteCustomTilingData& tilingData;

    uint32_t chunkElementNum;
    uint32_t chunkTileNum;
    uint32_t tileElementNum;
    uint32_t chunkTailElementNum;
    uint32_t processElementNum;

    uint32_t preCalcBufferSize;
    uint32_t bufferSize;

    // AscendC::GlobalTensor<float> workspaceGm;
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> workspaceInQueue;
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> workspaceOutQueue;
    AscendC::TBuf<> calcBuffer;
    AscendC::TBuf<> indicesBuffer;

    AscendC::GlobalTensor<T> inGm;
    AscendC::GlobalTensor<T> outGm;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
};

extern "C" __global__ __aicore__ void permute_custom(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(optiling::PermuteCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    AscendC::TPipe pipe;
    KernelPermute<DTYPE_INPUT> op(tilingData);
    AscendC::printf("start!!! aa 9 ---------------------------\n");
    op.Init(input, output, workspace);
    op.Process(pipe);
    AscendC::printf("end!!!\n");
}