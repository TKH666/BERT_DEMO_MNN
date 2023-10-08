//
// Created by 陈宇豪 on 2023/7/12.
//

#ifndef MNN_SUBMODEL_H
#define MNN_SUBMODEL_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "BERTEmbedding.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            class SubModel : public Express::Module {
            public:
                SubModel() = default;
                virtual ~SubModel() = default;

                virtual void loadParamByLayer(int layer, int startLayer = 0) = 0;
            };
        }
    }
}

#endif //MNN_SUBMODEL_H
