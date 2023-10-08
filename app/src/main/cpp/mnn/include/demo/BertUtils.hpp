//
// Created by 陈宇豪 on 2023/6/28.
//

#ifndef MNN_BERTUTILS_H
#define MNN_BERTUTILS_H
#include <MNN/expr/Module.hpp>
#include "BERT.hpp"

class BertUtils {
public:
    static void train(std::shared_ptr<MNN::Train::Model::BERTForClassification> model, std::string root);
};
#endif //MNN_BERTUTILS_H
