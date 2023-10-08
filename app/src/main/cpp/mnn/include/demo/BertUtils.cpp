//
// Created by 陈宇豪 on 2023/6/28.
//
#include <iostream>
#include "BertUtils.hpp"
#include "SGD.hpp"
#include "Conll2003Dataset.hpp"
#include "Transformer.hpp"
#include "Loss.hpp"
#include "ADAMW.hpp"
#include "LearningRateScheduler.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

void BertUtils::train(std::shared_ptr<Model::BERTForClassification> model, std::string root) {
    auto exe = Executor::getGlobalExecutor();
    BackendConfig config;
    exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 1);
    // exe->setGlobalExecutorConfig(MNN_FORWARD_AUTO, config, 1);
    std::shared_ptr<ADAMW> adamw(new ADAMW(model));
    adamw->setWeightDecay(0.01f);
    adamw->setLearningRate(2e-5f);
    adamw->setMomentum(0.9f);
    adamw->setMomentum2(0.999f);

//    std::shared_ptr<SGD> sgd(new SGD(model));
//    sgd->setMomentum(0.9f);
//    // sgd->setMomentum2(0.99f);
//    sgd->setWeightDecay(0.0005f);

    auto dataset = Conll2003Dataset::create(root, Conll2003Dataset::Mode::TRAIN);
    // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
    const size_t batchSize  = 1;
    const size_t numWorkers = 0;
    bool shuffle            = true;

    auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

    size_t iterations = dataLoader->iterNumber();

    auto testDataset            = Conll2003Dataset::create(root, Conll2003Dataset::Mode::TEST);
    const size_t testBatchSize  = 8;
    const size_t testNumWorkers = 0;
    shuffle                     = false;

    auto testDataLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, shuffle, testNumWorkers));

    size_t testIterations = testDataLoader->iterNumber();

    int totalEpochs = 10;
    int numTrainingSteps = iterations * totalEpochs;

    for (int epoch = 0; epoch < 50; ++epoch) {
        model->clearCache();
        exe->gc(Executor::FULL);
        exe->resetProfile();
        {
            AUTOTIME;
            dataLoader->reset();
            model->setIsTraining(true);
            Timer _100Time;

            int lastIndex = 0;
            int moveBatchSize = 0;
            for (int i = 0; i < iterations; i++) {
                // AUTOTIME;
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto label = example.second[0];
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Construct the attention mask
                std::vector<float> maskVec;
                std::vector<int> maskVecIdx;
                auto inputPtr = example.first[0]->readMap<int>();
                std::vector<int> dims = example.first[0]->getInfo()->dim;
                for (int i = 0; i < dims[0]; i++) {
                    for (int j = 0; j < dims[1]; j++) {
                        if (inputPtr[i * dims[1] + j] != 0) {
                            maskVec.push_back(1.0);
                            maskVecIdx.push_back(i * dims[1] + j);
                        } else {
                            maskVec.push_back(0.0);
                        }
                    }
                }
                auto attentionMask = _Const(maskVec.data(), example.first[0]->getInfo()->dim, NCHW, halide_type_of<float>());

                Timer _trainTime;
                auto logits = model->onForward({example.first[0], attentionMask})[0];
                auto ptr = logits->readMap<float>();
                MNN_PRINT("Forward Time: %f\n", (float)_trainTime.durationInUs() / 1000.0f);
                _trainTime.reset();

                // Compute Loss
                // TODO: Find the active loss, which can be optimized
                auto activeLoss = _Const(maskVecIdx.data(), {(int)maskVecIdx.size()}, NCHW, halide_type_of<int>());
                auto activeLogits = _GatherV2(_Reshape(logits, {-1, logits->getInfo()->dim[2]}), activeLoss, _Scalar<int>(0));
                auto activeLabels = _GatherV2(_Reshape(label, {-1}), activeLoss, _Scalar<int>(0));
                auto newActiveLabels = _OneHot(_Cast<int32_t>(activeLabels), _Scalar<int>(9), _Scalar<float>(1.0f), _Scalar<float>(0.0f));

                int ignoreIndex = -100;
                auto ignoredMask = _Cast<float>(_Unsqueeze(_NotEqual(activeLabels, _Scalar<int>(ignoreIndex)), {1}));
                newActiveLabels = newActiveLabels * ignoredMask;

                auto labelsPtr = newActiveLabels->readMap<float>();
                auto logitsPtr = activeLogits->readMap<float>();
                auto loss    = _CrossEntropy(_Softmax(activeLogits), newActiveLabels);
                auto lossPtr = loss->readMap<float>();
//                std::cout << " loss: " << loss->readMap<float>()[0] << std::endl;

                float rate = LrScheduler::linear(adamw->currentLearningRate(), epoch + 1, numTrainingSteps);
                // MNN_PRINT("Current learning rate: %.18f\n", rate);
                MNN_PRINT("Current Loss: %.8f\n", loss->readMap<float>()[0]);

                // float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                adamw->setLearningRate(rate);
                if (moveBatchSize % (10 * batchSize) == 0 || i == iterations - 1) {
                    std::cout << "epoch: " << (epoch);
                    std::cout << "  " << moveBatchSize << " / " << dataLoader->size();
                    std::cout << " loss: " << loss->readMap<float>()[0];
                    std::cout << " lr: " << rate;
                    std::cout << " time: " << (float)_100Time.durationInUs() / 1000.0f << " ms / " << (i - lastIndex) <<  " iter"  << std::endl;
                    std::cout.flush();
                    _100Time.reset();
                    lastIndex = i;
                }
                _trainTime.reset();
                adamw->step(loss);
                MNN_PRINT("Backward Time:%f\n", (float)_trainTime.durationInUs() / 1000.0f);
//                std::map<VARP, VARP> grad;
//                auto shape = loss->getInfo();
//                auto init= _Const(1.0f, shape->dim, shape->order);
//                adamw->backward(loss, example.first[0], init, grad);
            }
        }

//        int correct = 0;
//        testDataLoader->reset();
//        model->setIsTraining(false);
//        int moveBatchSize = 0;
//        for (int i = 0; i < testIterations; i++) {
//            auto data       = testDataLoader->next();
//            auto example    = data[0];
//            moveBatchSize += example.first[0]->getInfo()->dim[0];
//            if ((i + 1) % 100 == 0) {
//                std::cout << "test: " << moveBatchSize << " / " << testDataLoader->size() << std::endl;
//            }
//            auto cast       = _Cast<float>(example.first[0]);
//            example.first[0] = cast * _Const(1.0f / 255.0f);
//            auto predict    = model->forward(example.first[0]);
//            predict         = _ArgMax(predict, 1);
//            auto accu       = _Cast<int32_t>(_Equal(predict, _Cast<int32_t>(example.second[0]))).sum({});
//            correct += accu->readMap<int32_t>()[0];
//        }
//        auto accu = (float)correct / (float)testDataLoader->size();
//        std::cout << "epoch: " << epoch << "  accuracy: " << accu << std::endl;
        exe->dumpProfile();
    }
}