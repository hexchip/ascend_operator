#ifndef CUMSUM_CUSTIOM_TILING_DATA_H_
#define CUMSUM_CUSTIOM_TILING_DATA_H_

#include <cstdint>
#include <cstddef>

namespace optiling {
    static constexpr size_t MAX_HIERARCHY_LEVEL = 16;

    struct CumsumCustomTilingData
    {
        uint32_t col;
        uint32_t alignedRowElementNum;
        uint32_t redundantElementNum;
        uint32_t tileElementNum;
        uint32_t tileNum;
        uint32_t tailDataElementNum;
        uint32_t alignedTailDataElementNum;

        uint32_t cumsumTileElementNum;

        uint32_t cumsumTileHierarchy[MAX_HIERARCHY_LEVEL];
        uint32_t cumsumTotalTileElementNum;

        uint32_t tailDataCumsumTileHierarchy[MAX_HIERARCHY_LEVEL];

        uint32_t gatherMaskPatterns[16][8];
    };
}

#endif // CUMSUM_CUSTIOM_TILING_DATA_H_