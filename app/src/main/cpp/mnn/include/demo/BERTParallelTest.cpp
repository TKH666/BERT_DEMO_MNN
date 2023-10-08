//
// Created by 陈宇豪 on 2023/9/15.
//
#include "DemoUnit.hpp"
#include <iostream>
#include "SGD.hpp"
#include "BERTLayer.hpp"
#include "BertUtils.hpp"

#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace std;
using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

class BERTParallelSelfAttentionTest : public DemoUnit {
public:
    virtual int run(int argc, const char *argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out BERTParallelSelfAttentionTest /path/to/data/" << endl;
            return 0;
        }
        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        std::vector<MNNForwardType> types = {MNN_FORWARD_METAL, MNN_FORWARD_CPU};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_OPENCL};
        // std::vector<MNNForwardType> types = {MNN_FORWARD_CPU};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }

        // exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
        // exe->setGlobalExecutorConfig(MNN_FORWARD_METAL, config, 1);

        //TODO: TODO use here as flag to find them quickly
        auto bts = 128;
        auto heads = 12;

        auto encoder = std::make_shared<BERTLayer>(heads, 768, 3072, 0);
        std::string path = "../../pretrained_weights/bert/bert_1.mnn";
        auto params = Variable::load(path.c_str());
        encoder->loadParameters(params);
        auto keyParams = encoder->attention->self->key->parameters();
        auto valueParams = encoder->attention->self->value->parameters();
        auto queryParams = encoder->attention->self->query->parameters();

        //        auto x1 = _Const(1.0f, {1, 180, 768}, NCHW);
//        auto attentionMask1 = _Const(1.0f, {1, 1, 1, 180}, NCHW);
//        auto encoderOutput = encoder->attention->onForward({x1, attentionMask1})[0];

        auto selfAttn = std::make_shared<BERTSelfAttention>(768, heads, 0);
        selfAttn->key->loadParameters(keyParams);
        selfAttn->value->loadParameters(valueParams);
        selfAttn->query->loadParameters(queryParams);

        auto parallelSelfAttn = std::make_shared<ParallelSelfAttention>(768, heads, 0);
        parallelSelfAttn->key->loadParameters(keyParams);
        parallelSelfAttn->value->loadParameters(valueParams);
        parallelSelfAttn->query->loadParameters(queryParams);

        int repeatTime = 10;

        float noParallelSum = 0.0f, parallelSum = 0.0f;

        for (int k = 0; k < repeatTime; k++) {
            auto x2 = _Const(1.0f, {bts, 180, 768}, NCHW);
            auto attentionMask2 = _Const(1.0f, {bts, 1, 1, 180}, NCHW);
            // AUTOTIME;
            MNN::Timer _100Time;
            auto selfAttnOutput = selfAttn->onForward({x2, attentionMask2})[0];
            auto ptr = selfAttnOutput->readMap<float>();
            noParallelSum += (float)_100Time.durationInUs() / 1000.0f;
            // MNN_PRINT("No Parallel Time: %f ms\n", (float)_100Time.durationInUs() / 1000.0f);
            _100Time.reset();

            auto x3 = _Const(1.0f, {bts, 180, 768}, NCHW);
            auto attentionMask3 = _Const(1.0f, {bts, 1, 1, 180}, NCHW);

            MNN::Timer _100Time2;
            auto parallelSelfAttnOutput = parallelSelfAttn->onForward({x3, attentionMask3})[0];
            auto ptr2 = parallelSelfAttnOutput->readMap<float>();
            parallelSum += (float)_100Time2.durationInUs() / 1000.0f;
            // MNN_PRINT("Parallel Time: %f ms\n", (float)_100Time2.durationInUs() / 1000.0f);
            _100Time2.reset();

//            int diffCnt = 0;
//            auto shape = parallelSelfAttnOutput->getInfo();
//            for (int i = 0; i < shape->size; i++) {
//                if (fabs(ptr[i] - ptr2[i]) >= 1e-5) {
//                    MNN_PRINT("Error: %d %f %f\n", i, ptr[i], ptr2[i]);
//                    diffCnt++;
//                }
//            }
//            MNN_PRINT("Compuation error num: %d/%d\n", diffCnt, shape->size);
        }
        MNN_PRINT("No Parallel Average Time over %d tests: %f ms\n", repeatTime, noParallelSum / repeatTime);

        MNN_PRINT("Parallel Average Time over %d tests: %f ms\n", repeatTime, parallelSum / repeatTime);

        // Variable::save({parallelSelfAttnOutput}, "parallelSelfAttnOutput.mnn");

        MNN_PRINT("VARPSplitTest Success!\n");
    }
};

class BERTParallelTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out BERTParallelTest /path/to/data/" << endl;
            return 0;
        }

        std::string root = argv[1];

        int vocabSize = 30522;
        int numHiddenLayers = 12;
        int hiddenSize = 768;
        int intermediateSize = 3072;
        int numAttentionHeads = 12;
        float attnDropout = 0.0;
        float hiddenDropout = 0.0;
        std::shared_ptr<BERTForClassification> model(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout,9,
                                                                               true));

        int repeatTime = 1;
        // load the parameters
        auto bertEmbeddingParams = Variable::load("../../pretrained_weights/bert/bertEmbedding.snapshot.mnn");
//        for (auto& para : bertEmbeddingParams) {
//            para.fix(VARP::TRAINABLE);
//        }
        model->bert->embedding->loadParameters(bertEmbeddingParams);

        auto bertEncoderParams = Variable::load("../../pretrained_weights/bert/bertEncoder.snapshot.mnn");
        for (auto& para : bertEncoderParams) {
            para.fix(VARP::TRAINABLE);
        }
        model->bert->encoder->loadParameters(bertEncoderParams);

        auto bertPoolerParams = Variable::load("../../pretrained_weights/bert/bertPooler.snapshot.mnn");
        for (auto& para : bertPoolerParams) {
            para.fix(VARP::TRAINABLE);
        }
        model->bert->pooler->loadParameters(bertPoolerParams);

        BertUtils::train(model, root);
    }
};

class LoadMultipleBackendTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out LoadMultipleBackendTest /path/to/data/" << endl;
            return 0;
        }

        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;

        std::vector<MNNForwardType> types = {MNN_FORWARD_CPU, MNN_FORWARD_OPENCL, MNN_FORWARD_METAL};
        for (auto& type : types) {
            exe->setGlobalExecutorConfig(type, config, 1);
        }
        // exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 1);
        // exe->setGlobalExecutorConfig(MNN_FORWARD_METAL, config, 1);
    }
};

DemoUnitSetRegister(BERTParallelSelfAttentionTest, "BERTParallelSelfAttentionTest");
DemoUnitSetRegister(BERTParallelTest, "BERTParallelTest");
DemoUnitSetRegister(LoadMultipleBackendTest, "LoadMultipleBackendTest");