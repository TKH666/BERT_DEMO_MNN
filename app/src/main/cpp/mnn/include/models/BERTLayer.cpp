//
// Created by 陈宇豪 on 2023/6/16.
//
#include "BERTLayer.hpp"
#include "MLayerNorm.hpp"
#include <thread>
#include "core/Concurrency.h"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            BERTSelfAttention::BERTSelfAttention(int hiddenSize, int numAttentionHeads,
                                                 float attentionProbsDropoutProb, bool forParallel) {
                this->numAttentionHeads = numAttentionHeads;
                this->attentionHeadSize = hiddenSize / numAttentionHeads;
                this->allHeadSize = numAttentionHeads * attentionHeadSize;

                this->forParallel = forParallel;

                query.reset(NN::Linear(hiddenSize, allHeadSize, true));
                query->setName("query");

                key.reset(NN::Linear(hiddenSize, allHeadSize, true));
                key->setName("key");

                value.reset(NN::Linear(hiddenSize, allHeadSize, true));
                value->setName("value");

                if (forParallel) {
                    parallelQuery.resize(numAttentionHeads);
                    parallelKey.resize(numAttentionHeads);
                    parallelValue.resize(numAttentionHeads);

                    for (int i = 0; i < numAttentionHeads; i++) {
                        parallelQuery[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                        parallelQuery[i]->setName("query" + std::to_string(i));

                        parallelKey[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                        parallelKey[i]->setName("key" + std::to_string(i));

                        parallelValue[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                        parallelValue[i]->setName("value" + std::to_string(i));
                    }
                }

                dropout.reset(NN::Dropout(attentionProbsDropoutProb));
                registerModel({query, key, value, dropout});
            }

            std::vector<Express::VARP> BERTSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
                std::call_once(mOnceFlag, [&]() {
                    if (this->forParallel){
                        auto keyParams = key->parameters();
                        auto queryParams = query->parameters();
                        auto valueParams = value->parameters();

                        auto queryBiasParams = _Split(queryParams[0], {numAttentionHeads}, 1);
                        auto queryWeightParams = _Split(queryParams[1], {numAttentionHeads}, 0);
                        auto keyBiasParams = _Split(keyParams[0], {numAttentionHeads}, 1);
                        auto keyWeightParams = _Split(keyParams[1], {numAttentionHeads}, 0);
                        auto valueBiasParams = _Split(valueParams[0], {numAttentionHeads}, 1);
                        auto valueWeightParams = _Split(valueParams[1], {numAttentionHeads}, 0);

                        for (int i = 0; i < numAttentionHeads; i++) {
                            parallelKey[i]->loadParameters({_Clone(keyBiasParams[i], true), _Clone(keyWeightParams[i], true)});
                            parallelQuery[i]->loadParameters({_Clone(queryBiasParams[i], true), _Clone(queryWeightParams[i], true)});
                            parallelValue[i]->loadParameters({_Clone(valueBiasParams[i], true), _Clone(valueWeightParams[i], true)});
                        }

                        //TODO modify register model
                        deleteModel({query, key, value});
                        registerModel(parallelQuery);
                        registerModel(parallelKey);
                        registerModel(parallelValue);
                    }
                });
                using namespace Express;
                VARP x = inputs[0];
                VARP attentionMask = inputs[1];

                if (!this->forParallel) {
                    auto mixedQueryLayer = query->forward(x);
                    auto mixedKeyLayer = key->forward(x);
                    auto mixedValueLayer = value->forward(x);

                    auto shape = mixedKeyLayer->getInfo()->dim;
                    auto queryLayer = _Permute(_Reshape(mixedQueryLayer, {shape[0], shape[1], numAttentionHeads, attentionHeadSize}), {0, 2, 1, 3});
                    auto keyLayer = _Permute(_Reshape(mixedKeyLayer, {shape[0], shape[1], numAttentionHeads, attentionHeadSize}), {0, 2, 1, 3});
                    auto valueLayer = _Permute(_Reshape(mixedValueLayer, {shape[0], shape[1], numAttentionHeads, attentionHeadSize}), {0, 2, 1, 3});

                    auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 1, 3, 2}));
                    attentionScores = _Divide(attentionScores, _Const((float) sqrt(attentionHeadSize), {}, NCHW));
                    attentionScores = attentionScores + attentionMask;

                    // Normalize the attention scores to probabilities.
                    auto attentionProbs = _Softmax(attentionScores, -1);

                    // TODO: This is actually dropping out entire tokens to attend to, which might
                    //  seem a bit unusual, but is taken from the original Transformer paper.
                    attentionProbs = dropout->forward(attentionProbs);

                    auto contextLayer = _MatMul(attentionProbs, valueLayer);
                    contextLayer = _Permute(contextLayer, {0, 2, 1, 3});
                    auto newContextLayerShape = contextLayer->getInfo()->dim;
                    contextLayer = _Reshape(contextLayer, {newContextLayerShape[0], newContextLayerShape[1], allHeadSize});

                    return {contextLayer};
                }
                else{
                    attentionMask = _Squeeze(attentionMask, {1});

                    std::vector<VARP> allReduced;

                    for (int i = 0; i < numAttentionHeads; ++i) {
                        auto xClone = _Clone(x, true); // 在这里会调用readMap
                        auto queryLayer = parallelKey[i]->forward(xClone);
                        auto keyLayer = parallelKey[i]->forward(xClone);
                        auto valueLayer = parallelValue[i]->forward(xClone);

                        auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));
                        attentionScores = attentionScores + _Clone(attentionMask, true);

                        // Normalize the attention scores to probabilities.
                        auto attentionProbs = _Softmax(attentionScores, -1);

                        attentionProbs = dropout->forward(attentionProbs);

                        auto contextLayer = _MatMul(attentionProbs, valueLayer);

                        allReduced.emplace_back(_Unsqueeze(contextLayer, {1}));
                    };

                    Variable::prepareComputeParallel(allReduced);
                    // Variable::prepareCompute(allReduced);

                    auto contextLayers = _Concat(allReduced, 1);
                    contextLayers = _Permute(contextLayers, {0, 2, 1, 3});
                    auto newContextLayerShape = contextLayers->getInfo()->dim;
                    contextLayers = _Reshape(contextLayers, {newContextLayerShape[0], newContextLayerShape[1], allHeadSize});
                    return {contextLayers};
                }
            }

            ParallelSelfAttention::ParallelSelfAttention(int hiddenSize, int numAttentionHeads,
                                                 float attentionProbsDropoutProb) {
                this->numAttentionHeads = numAttentionHeads;
                this->attentionHeadSize = hiddenSize / numAttentionHeads;
                this->allHeadSize = numAttentionHeads * attentionHeadSize;

                query.reset(NN::Linear(hiddenSize, allHeadSize, true));
                key.reset(NN::Linear(hiddenSize, allHeadSize, true));
                value.reset(NN::Linear(hiddenSize, allHeadSize, true));

                parallelQuery.resize(numAttentionHeads);
                parallelKey.resize(numAttentionHeads);
                parallelValue.resize(numAttentionHeads);

                for (int i = 0; i < numAttentionHeads; i++) {
                    parallelQuery[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelQuery[i]->setName("query" + std::to_string(i));

                    parallelKey[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelKey[i]->setName("key" + std::to_string(i));

                    parallelValue[i].reset(NN::Linear(hiddenSize, attentionHeadSize, true));
                    parallelValue[i]->setName("value" + std::to_string(i));
                }

                this->allHeadSize = numAttentionHeads * attentionHeadSize;

                dropout.reset(NN::Dropout(attentionProbsDropoutProb));

                // multiple backends
                auto exe = Executor::getGlobalExecutor();
                auto availableBackends = exe->getAvailableBackends();
                if (availableBackends.size() > 1) {
                    concatKey.reset(NN::Linear(hiddenSize, attentionHeadSize * 6, true));
                    concatQuery.reset(NN::Linear(hiddenSize, attentionHeadSize * 6, true));
                    concatValue.reset(NN::Linear(hiddenSize, attentionHeadSize * 6, true));
                    registerModel({concatKey, concatQuery, concatValue});
                    registerModel(std::vector<std::shared_ptr<Module>>(parallelKey.begin() + numAttentionHeads / 2 - 1, parallelKey.end()));
                    registerModel(std::vector<std::shared_ptr<Module>>(parallelQuery.begin() + numAttentionHeads / 2 - 1, parallelQuery.end()));
                    registerModel(std::vector<std::shared_ptr<Module>>(parallelValue.begin() + numAttentionHeads / 2 - 1, parallelValue.end()));
                } else {
                    registerModel(parallelKey);
                    registerModel(parallelQuery);
                    registerModel(parallelValue);
                }

                //registerModel({query, key, value, dropout});
//                registerModel(parallelQuery);
//                registerModel(parallelKey);
//                registerModel(parallelValue);
                registerModel({dropout});
            }

            std::vector<Express::VARP> ParallelSelfAttention::onForward(const std::vector<Express::VARP> &inputs) {
                std::call_once(mOnceFlag, [&]() {
                    auto keyParams = key->parameters();
                    auto queryParams = query->parameters();
                    auto valueParams = value->parameters();

                    auto queryBiasParams = _Split(queryParams[0], {numAttentionHeads}, 1);
                    auto queryWeightParams = _Split(queryParams[1], {numAttentionHeads}, 0);
                    auto keyBiasParams = _Split(keyParams[0], {numAttentionHeads}, 1);
                    auto keyWeightParams = _Split(keyParams[1], {numAttentionHeads}, 0);
                    auto valueBiasParams = _Split(valueParams[0], {numAttentionHeads}, 1);
                    auto valueWeightParams = _Split(valueParams[1], {numAttentionHeads}, 0);

                    for (int i = 0; i < numAttentionHeads; i++) {
                        parallelKey[i]->loadParameters({_Clone(keyBiasParams[i], true), _Clone(keyWeightParams[i], true)});
                        parallelQuery[i]->loadParameters({_Clone(queryBiasParams[i], true), _Clone(queryWeightParams[i], true)});
                        parallelValue[i]->loadParameters({_Clone(valueBiasParams[i], true), _Clone(valueWeightParams[i], true)});
                    }

                    auto exe = Executor::getGlobalExecutor();
                    auto availableBackends = exe->getAvailableBackends();
                    if (availableBackends.size() > 1) {
                        auto concatKeyBiasParams = _Concat(std::vector<VARP>(keyBiasParams.begin(), keyBiasParams.begin() + numAttentionHeads / 2), 1);
                        auto concatKeyWeightParams = _Concat(std::vector<VARP>(keyWeightParams.begin(), keyWeightParams.begin() + numAttentionHeads / 2), 0);
                        concatKey->loadParameters({_Clone(concatKeyBiasParams, true), _Clone(concatKeyWeightParams, true)});

                        auto concatQueryBiasParams = _Concat(std::vector<VARP>(queryBiasParams.begin(), queryBiasParams.begin() + numAttentionHeads / 2), 1);
                        auto concatQueryWeightParams = _Concat(std::vector<VARP>(queryWeightParams.begin(), queryWeightParams.begin() + numAttentionHeads / 2), 0);
                        concatQuery->loadParameters({_Clone(concatQueryBiasParams, true), _Clone(concatQueryWeightParams, true)});

                        auto concatValueBiasParams = _Concat(std::vector<VARP>(valueBiasParams.begin(), valueBiasParams.begin() + numAttentionHeads / 2), 1);
                        auto concatValueWeightParams = _Concat(std::vector<VARP>(valueWeightParams.begin(), valueWeightParams.begin() + numAttentionHeads / 2), 0);
                        concatValue->loadParameters({_Clone(concatValueBiasParams, true), _Clone(concatValueWeightParams, true)});
                    }
                });

                using namespace Express;
                VARP x = inputs[0];
                VARP attentionMask = inputs[1];

                // squeeze the attentionMask
                attentionMask = _Squeeze(attentionMask, {1});

                std::vector<VARP> allReduced;

                auto exe = Executor::getGlobalExecutor();
                auto availableBackends = exe->getAvailableBackends();

                // TODO: Only support two backends now
                if (availableBackends.size() > 1) {
                    MNN_PRINT("Using multiple backends\n");
                    std::vector<VARP> multipleAllReduced;
                    // concat part
                    auto concatXClone = _Clone(x, true); // 在这里会调用readMap
                    auto concatQueryLayer = concatQuery->forward(concatXClone);
                    auto concatKeyLayer = concatKey->forward(concatXClone);
                    auto concatValueLayer = concatValue->forward(concatXClone);

                    auto concatAttentionScores = _MatMul(concatQueryLayer, _Transpose(concatKeyLayer, {0, 2, 1}));
                    concatAttentionScores = concatAttentionScores + _Clone(attentionMask, true);

                    // Normalize the attention scores to probabilities.
                    auto concatAttentionProbs = _Softmax(concatAttentionScores, -1);

                    concatAttentionProbs = dropout->forward(concatAttentionProbs);

                    auto concatContextLayer = _MatMul(concatAttentionProbs, concatValueLayer);
                    multipleAllReduced.emplace_back(concatContextLayer);
//                    auto shape = concatContextLayer->getInfo()->dim;
//                    multipleAllReduced.emplace_back(_Reshape(_Unsqueeze(concatContextLayer, {1}), {shape[0], numAttentionHeads / 2, shape[1], shape[2] / (numAttentionHeads / 2)}));

                    for (int i = numAttentionHeads / 2; i < numAttentionHeads; ++i) {
                        auto xClone = _Clone(x, true);
                        auto queryLayer = parallelKey[i]->forward(xClone);
                        auto keyLayer = parallelKey[i]->forward(xClone);
                        auto valueLayer = parallelValue[i]->forward(xClone);

                        auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));
                        attentionScores = attentionScores + _Clone(attentionMask, true);

                        // Normalize the attention scores to probabilities.
                        auto attentionProbs = _Softmax(attentionScores, -1);

                        attentionProbs = dropout->forward(attentionProbs);

                        auto contextLayer = _MatMul(attentionProbs, valueLayer);

                        multipleAllReduced.emplace_back(_Unsqueeze(contextLayer, {1}));
                    }
                    Variable::prepareComputeParallel(multipleAllReduced, true);

                    // split the concatContextLayer
                    auto splitedConcatContextLayer = _Split(concatContextLayer, {numAttentionHeads / 2}, 2);
                    for (int i = 0; i < numAttentionHeads / 2; ++i) {
                        allReduced.emplace_back(_Unsqueeze(splitedConcatContextLayer[i], {1}));
                    }
                    for (int i = 1; i < multipleAllReduced.size(); ++i) {
                        allReduced.emplace_back(multipleAllReduced[i]);
                    }
                    // Variable::prepareComputeParallel(allReduced, true);
                } else {
                    for (int i = 0; i < numAttentionHeads; ++i) {
                        auto xClone = _Clone(x, true); // 在这里会调用readMap
                        auto queryLayer = parallelKey[i]->forward(xClone);
                        auto keyLayer = parallelKey[i]->forward(xClone);
                        auto valueLayer = parallelValue[i]->forward(xClone);

                        auto attentionScores = _MatMul(queryLayer, _Transpose(keyLayer, {0, 2, 1}));
                        attentionScores = attentionScores + _Clone(attentionMask, true);

                        // Normalize the attention scores to probabilities.
                        auto attentionProbs = _Softmax(attentionScores, -1);

                        attentionProbs = dropout->forward(attentionProbs);

                        auto contextLayer = _MatMul(attentionProbs, valueLayer);

                        allReduced.emplace_back(_Unsqueeze(contextLayer, {1}));
                    }
                    Variable::prepareComputeParallel(allReduced);
                }

                // Variable::prepareCompute(allReduced);

                auto contextLayers = _Concat(allReduced, 1);
                contextLayers = _Permute(contextLayers, {0, 2, 1, 3});
                auto newContextLayerShape = contextLayers->getInfo()->dim;
                contextLayers = _Reshape(contextLayers, {newContextLayerShape[0], newContextLayerShape[1], allHeadSize});
                return {contextLayers};
            }

            BERTIntermediate::BERTIntermediate(int hiddenSize, int intermediateSize) {
                dense.reset(NN::Linear(hiddenSize, intermediateSize, true));
                registerModel({dense});
            }

            std::vector<Express::VARP> BERTIntermediate::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];

                auto hiddenStates = dense->forward(x);
                // auto ptr = dense->parameters()[0]->readMap<float>();
                // Calculate Gelu with Phi function
                auto geluPhi = _Const(0.5f) * (_Const(1.0f) + _Erf(hiddenStates / _Const(sqrt(2.0))));
                hiddenStates = hiddenStates * geluPhi;
                // hiddenStates = _Gelu(hiddenStates);

                return {hiddenStates};
            }


            BERTOutput::BERTOutput(int hiddenSize, int intermediateSize, float dropoutProb) {
                dense.reset(NN::Linear(intermediateSize, hiddenSize, true));
                layerNorm.reset(new MLayerNorm({hiddenSize}, true, 1e-12));
                dropout.reset(NN::Dropout(dropoutProb));

                registerModel({dense, layerNorm, dropout});
            }

            std::vector<Express::VARP> BERTOutput::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                VARP inputTensor = inputs[1];

                auto hiddenStates = dense->forward(x);
                hiddenStates = dropout->forward(hiddenStates);
                hiddenStates = layerNorm->forward(hiddenStates + inputTensor);

                return {hiddenStates};
            }

            BERTSelfOutput::BERTSelfOutput(int hiddenSize, float dropoutProb) {
                dense.reset(NN::Linear(hiddenSize, hiddenSize, true));
                layerNorm.reset(new MLayerNorm({hiddenSize}, true, 1e-12));
                dropout.reset(NN::Dropout(dropoutProb));

                registerModel({dense, layerNorm, dropout});
            }

            std::vector<Express::VARP> BERTSelfOutput::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];
                VARP inputTensor = inputs[1];

                auto hiddenStates = dense->forward(x);
                hiddenStates = dropout->forward(hiddenStates);
                hiddenStates = layerNorm->forward(hiddenStates + inputTensor);

                return {hiddenStates};
            }

            BERTAttention::BERTAttention(int hiddenSize, int numAttentionHeads, float dropoutProb, bool forParallel) {
                self.reset(new BERTSelfAttention(hiddenSize, numAttentionHeads, dropoutProb, forParallel));
                output.reset(new BERTSelfOutput(hiddenSize, dropoutProb));

                registerModel({self, output});
            }

            std::vector<Express::VARP> BERTAttention::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP inputTensor = inputs[0];
                VARP attentionMask = inputs[1];

                auto selfOutput = self->onForward({inputTensor, attentionMask});
                // auto ptr = selfOutput[0]->readMap<float>();
                auto attentionOutput = output->onForward({selfOutput[0], inputTensor});
                return {attentionOutput};
            }

            BERTLayer::BERTLayer(int numAttentionHeads, int hiddenSize, int intermediateSize, float dropoutProb, bool forParallel) {
                attention.reset(new BERTAttention(hiddenSize, numAttentionHeads, dropoutProb, forParallel));
                intermediate.reset(new BERTIntermediate(hiddenSize, intermediateSize));
                output.reset(new BERTOutput(hiddenSize, intermediateSize, dropoutProb));

                registerModel({attention, intermediate, output});
            }

            std::vector<Express::VARP> BERTLayer::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP hiddenStates = inputs[0];
                VARP attentionMask = inputs[1];

                auto attentionOutput = attention->onForward({hiddenStates, attentionMask});
                auto intermediateOutput = intermediate->onForward({attentionOutput[0]});
                auto layerOutput = output->onForward({intermediateOutput[0], attentionOutput[0]});
                return {layerOutput[0]};
            }
        }
    }
}