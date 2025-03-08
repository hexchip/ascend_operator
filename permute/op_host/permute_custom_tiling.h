#pragma once

#include "graph/graph.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
    ge::graphStatus PermuteCustomTilingFunc(gert::TilingContext* context);
}
