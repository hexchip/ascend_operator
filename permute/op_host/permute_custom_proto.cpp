#include <vector>

#include "register/op_impl_registry.h"

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

    IMPL_OP(PermuteCustom).InferShape(InferShape).InferDataType(InferDataType);
}


