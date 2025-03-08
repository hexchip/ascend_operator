#include <cstdint>
#include <cstddef>

namespace optiling {
    static constexpr size_t MAX_DIM_NUM = 25;

    struct PermuteCustomTilingData
    {
        uint8_t shapeDim;
        uint32_t targetShape[MAX_DIM_NUM];
        uint32_t perm[MAX_DIM_NUM];
        uint32_t originShapeStrides[MAX_DIM_NUM];

        uint64_t tileElementNum;
        uint64_t inputTailDataBlockNum;
        uint64_t smallChunkElementNum;
        uint64_t smallChunkTileNum;
        uint64_t smallChunkTailElementNum;
        uint64_t bigChunkElementNum;
        uint64_t bigChunkTileNum;
        uint64_t bigChunkTailElementNum;
    };
}