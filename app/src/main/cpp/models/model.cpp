//
// Created by 陈宇豪 on 2023/1/10.
//

#include "model.h"
#include "MobileNetV2.h"
#include "MobileViT.h"
#include "BERT.h"
#include "log.h"

using namespace MNN;
using namespace MNN::Train::Model;

namespace FTPipeHD {
    std::shared_ptr<Express::Module> ModelZoo::modelPtr = nullptr;
    std::shared_ptr<SubModel> ModelZoo::subModelPtr = nullptr;

    void ModelZoo::createModel(std::string &modelName, std::map<std::string, double> &modelArgs) {
        if (modelName == "BERTForClassification") {
            int vocabSize = (int) modelArgs["vocab_size"];
            int numHiddenLayers = (int) modelArgs["num_hidden_layers"];
            int hiddenSize = (int) modelArgs["hidden_size"];
            int intermediateSize = (int) modelArgs["intermediate_size"];
            int numAttentionHeads = (int) modelArgs["num_attention_heads"];
            float attnDropout = (float) modelArgs["attention_dropout_prob"];
            float hiddenDropout = (float) modelArgs["hidden_dropout_prob"];

            modelPtr = std::shared_ptr<BERTForClassification>(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9));
        }
    }
}