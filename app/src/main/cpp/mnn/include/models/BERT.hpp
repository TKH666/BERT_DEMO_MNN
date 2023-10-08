//
// Created by 陈宇豪 on 2023/6/12.
//

#ifndef MNN_BERT_H
#define MNN_BERT_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include "BERTEmbedding.hpp"
#include "SubModel.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            class BERTEncoder : public Module {
            public:
                BERTEncoder(int numHiddenLayers,  int numAttentionHeads, int hiddenSize, int intermediateSize, float dropoutProb = 0.1, bool forParallel = false);
                std::vector<std::shared_ptr<Module>> layer;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class BERTPooler : public Module {
            public:
                BERTPooler(int hiddenSize);
                std::shared_ptr<Module> dense;
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
            };

            class MNN_PUBLIC BERT : public Express::Module {
            public:
                BERT(int vocabSize, int hidden = 768, int nLayers = 12, int attnHeads = 12, int intermediateSize = 3072, float dropout = 0.1, bool forParallel = false);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

                std::shared_ptr<BERTEmbedding> embedding;
                std::shared_ptr<BERTEncoder> encoder;
                std::shared_ptr<BERTPooler> pooler;
            };

            class MNN_PUBLIC BERTForClassification : public Express::Module {
            public:
                BERTForClassification(int vocabSize, int hidden = 768, int nLayers = 12, int attnHeads = 12, int intermediateSize = 3072, float attentionDropout = 0.1, float hiddenDropout = 0.1, int numClasses = 9, bool forParallel=false);
                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

                std::shared_ptr<BERT> bert;
                std::shared_ptr<Module> dropout;
                std::shared_ptr<Module> classifier;

                int nLayers;
            };

            class SubBERTForClassification : public SubModel {
            public:
                SubBERTForClassification(int start, int end, std::map<std::string, double>& args);

                virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

                virtual void loadParamByLayer(int layer, int startLayer = 1) override;

                std::shared_ptr<BERTEmbedding> embedding;
                std::vector<std::shared_ptr<Module> > encoders;
                std::shared_ptr<BERTPooler> pooler;
                std::shared_ptr<Module> dropout;
                std::shared_ptr<Module> classifier;
            private:
                int numHiddenLayers;
                int totalLayers;
            };
        }
    }
}

#endif //MNN_BERT_H
