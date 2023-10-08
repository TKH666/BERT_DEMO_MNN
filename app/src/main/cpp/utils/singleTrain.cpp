//
// Created by 陈宇豪 on 2023/6/5.
//

#include "singleTrain.h"
#include "MNN/expr/Module.hpp"
#include "NN.hpp"
#include "model.h"
#include "SGD.hpp"
#include "Loss.hpp"
#include "LearningRateScheduler.hpp"
#include "log.h"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include "datasets.h"
#include "commonStates.h"

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::Train;

namespace FTPipeHD {
    void initTrain(int batchSize) {
        CommonStates::setBatchSize(batchSize);
        auto exe = Executor::getGlobalExecutor();
        BackendConfig config;
        // exe->setGlobalExecutorConfig(MNN_FORWARD_VULKAN, config, 4);
        // exe->setGlobalExecutorConfig(MNN_FORWARD_OPENCL, config, 4);
        exe->setGlobalExecutorConfig(MNN_FORWARD_CPU, config, 4);
    }

    void singleTrainOneEpoch(int epoch) {

        auto model = ModelZoo::modelPtr;
        auto exe = Executor::getGlobalExecutor();

        auto dataLoader = Datasets::trainSetLoader;
        size_t iterations = dataLoader->iterNumber();

        std::shared_ptr<SGD> sgd(new SGD(model));
        sgd->setMomentum(0.9f);
        sgd->setWeightDecay(0.0005f);

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
            for (int i = 0; i < 1; i++) {
                // AUTOTIME
                auto trainData  = dataLoader->next();
                auto example    = trainData[0];
                auto label = example.second[0];
                moveBatchSize += example.first[0]->getInfo()->dim[0];

                // Construct the attention mask
                std::vector<float> maskVec;
                std::vector<int> maskVecIdx;
                auto inputPtr = example.first[0]->readMap<int>();
                std::vector<int> dims = example.first[0]->getInfo()->dim;
                for (int s = 0; s < dims[0]; s++) {
                    for (int j = 0; j < dims[1]; j++) {
                        if (inputPtr[s * dims[1] + j] != 0) {
                            maskVec.push_back(1.0);
                            maskVecIdx.push_back(s * dims[1] + j);
                        } else {
                            maskVec.push_back(0.0);
                        }
                    }
                }
                auto attentionMask = _Const(maskVec.data(), example.first[0]->getInfo()->dim, NCHW, halide_type_of<float>());

                MNN::Timer _100Time;
                auto logits = model->onForward({example.first[0], attentionMask})[0];

                // Compute Loss
                // TODO: Find the active loss, which can be optimized
                auto activeLoss = _Const(maskVecIdx.data(), {(int)maskVecIdx.size()}, NCHW, halide_type_of<int>());
                auto activeLogits = _GatherV2(_Reshape(logits, {-1, logits->getInfo()->dim[2]}), activeLoss, _Scalar<int>(0));
                auto activeLabels = _GatherV2(_Reshape(label, {-1}), activeLoss, _Scalar<int>(0));
                auto newActiveLabels = _OneHot(_Cast<int32_t>(activeLabels), _Scalar<int>(9), _Scalar<float>(1.0f), _Scalar<float>(0.0f));

                int ignoreIndex = -100;
                auto ignoredMask = _Cast<float>(_Unsqueeze(_NotEqual(activeLabels, _Scalar<int>(ignoreIndex)), {1}));
                newActiveLabels = newActiveLabels * ignoredMask;

                auto loss    = _CrossEntropy(_Softmax(activeLogits), newActiveLabels);
                auto lossPtr = loss->readMap<float>();
                LOGI("Forwarding time:%f, Loss: %f\n", (float)_100Time.durationInUs() / 1000.0f, lossPtr[0]);

                float rate   = LrScheduler::inv(0.01, epoch * iterations + i, 0.0001, 0.75);
                sgd->setLearningRate(rate);

                // sgd->step(loss);
            }
        }

        exe->dumpProfile();
    }
}