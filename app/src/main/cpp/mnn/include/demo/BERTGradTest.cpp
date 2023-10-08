//
// Created by 陈宇豪 on 2023/8/8.
//
#include "DemoUnit.hpp"
#include <iostream>
#include "Conll2003Dataset.hpp"
#include "SGD.hpp"
#include "BERTEmbedding.hpp"
#include "MEmbedding.hpp"

using namespace std;
using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class EmbeddingGradTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out EmbeddingGradTest /path/to/data/" << endl;
            return 0;
        }

        std::string vocabPath = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/vocab_files/bert-base-uncased-vocab.txt";
        std::string root = argv[1];

        auto dataset = Conll2003Dataset::create(root, Conll2003Dataset::Mode::TRAIN);
        const size_t batchSize  = 1;
        const size_t numWorkers = 0;
        bool shuffle            = true;

        auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

        dataLoader->reset();

        auto trainData  = dataLoader->next();
        auto example    = trainData[0];
        auto label = example.second[0];

        auto x = example.first[0];
        std::vector<int> allZeros(x->getInfo()->size, 0);
        VARP tokenTypeIds = _Const(allZeros.data(), x->getInfo()->dim, NCHW, halide_type_of<int>());

        int vocabSize = 30522;
        int hiddenSize = 768;

        // Embedding Check
        std::shared_ptr<Module> m(new BERTEmbedding(vocabSize, hiddenSize, 512, 2, 0.0));
        std::shared_ptr<Module> wordEmbeddings = std::make_shared<MEmbedding>(vocabSize, hiddenSize);
        std::shared_ptr<SGD> sgd(new SGD(wordEmbeddings));

        auto out = wordEmbeddings->onForward({x, tokenTypeIds})[0];
        auto ptr = out->readMap<float>();

        auto shape = out->getInfo();
        auto init= _Const(1.0f, shape->dim, shape->order);

        std::map<VARP, VARP> grad;
        sgd->backward(out, x, init, grad);
        MNN_PRINT("Embedding Grad Test Finish!\n");
    }
};



DemoUnitSetRegister(EmbeddingGradTest, "EmbeddingGradTest");
