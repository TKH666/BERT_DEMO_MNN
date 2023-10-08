//
// Created by 陈宇豪 on 2023/6/12.
//
#include "BERT.h"
#include "BERTLayer.h"
#include "MNN/AutoTime.hpp"
#include <memory>

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            BERTPooler::BERTPooler(int hiddenSize) {
                dense.reset(NN::Linear(hiddenSize, hiddenSize, true));
                registerModel({dense});
            }

            std::vector<Express::VARP>
            BERTPooler::onForward(const std::vector<Express::VARP> &inputs) {
                // We "pool" the model by simply taking the hidden state corresponding
                // to the first token.
                using namespace Express;
                VARP hiddenStates = inputs[0];

                int startSlice[] = {0, 0, 0};
                int sizeSlice[] = {-1, 1, -1};

                auto firstTokenTensor = _Slice(hiddenStates,
                                               _Const(startSlice, {3}, NCHW, halide_type_of<int>()),
                                               _Const(sizeSlice, {3}, NCHW,
                                                      halide_type_of<int>()));
                auto pooledOutput = dense->forward(firstTokenTensor);
                pooledOutput = _Tanh(pooledOutput);

                return {pooledOutput};
            }


            BERTEncoder::BERTEncoder(int numHiddenLayers, int numAttentionHeads, int hiddenSize,
                                     int intermediateSize, float dropoutProb, bool forParallel) {
                for (int i = 0; i < numHiddenLayers; i++) {
                    layer.push_back(std::make_shared<BERTLayer>(numAttentionHeads, hiddenSize,
                                                                intermediateSize, dropoutProb,
                                                                forParallel));
                }

                registerModel(layer);
            }

            std::vector<Express::VARP>
            BERTEncoder::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP hiddenStates = inputs[0];
                VARP attentionMask = inputs[1];

                // TODO: outputAllEncodedLayers is ignored here
                for (int i = 0; i < layer.size(); i++) {
                    MNN::Timer _100Time;
                    hiddenStates = layer[i]->onForward({hiddenStates, attentionMask})[0];
                    // auto x = hiddenStates->readMap<float>();
                    // MNN_PRINT("Encoder%d time:%f\n", i, (float)_100Time.durationInUs() / 1000.0f);
                    _100Time.reset();
                }

                return {hiddenStates};
            }

            BERT::BERT(int vocabSize, int hidden, int nLayers, int attnHeads, int intermediateSize,
                       float dropout, bool forParallel) {
                this->embedding = std::make_shared<BERTEmbedding>(vocabSize, hidden, 512, 2,
                                                                  dropout);
                this->encoder = std::make_shared<BERTEncoder>(nLayers, attnHeads, hidden,
                                                              intermediateSize, dropout,
                                                              forParallel);
                this->pooler = std::make_shared<BERTPooler>(hidden);

                registerModel({embedding, encoder, pooler});
            }

            std::vector<Express::VARP> BERT::onForward(const std::vector<Express::VARP> &inputs) {
                // {inputsIds, tokenTypeIds, attentionMask, outputAllEncodedLayers}
                VARP inputIds = inputs[0];
                VARP tokenTypeIds;
                VARP attentionMask;

                if (inputs.size() < 2) {
                    // attentionMask is None
                    std::vector<float> maskVec;
                    auto inputPtr = inputIds->readMap<int>();
                    std::vector<int> dims = inputIds->getInfo()->dim;
                    for (int i = 0; i < dims[0]; i++) {
                        for (int j = 0; j < dims[1]; j++) {
                            if (inputPtr[i * dims[1] + j] != 0) {
                                maskVec.push_back(1.0);
                            } else {
                                maskVec.push_back(0.0);
                            }
                        }
                    }
                    attentionMask = _Const(maskVec.data(), inputIds->getInfo()->dim, NCHW,
                                           halide_type_of<float>());
                } else {
                    attentionMask = inputs[1];
                }

                if (inputs.size() < 3) {
                    // tokenTypeIds is None
                    std::vector<int> allZeros(inputIds->getInfo()->size, 0);
                    tokenTypeIds = _Const(allZeros.data(), inputIds->getInfo()->dim, NCHW,
                                          halide_type_of<int>());
                } else {
                    tokenTypeIds = inputs[2];
                }

                auto extendedAttentionMask = _Unsqueeze(attentionMask, {1, 2});
                extendedAttentionMask =
                        (_Scalar<float>(1.0) - extendedAttentionMask) * _Scalar<float>(-10000.0);

                auto embeddingOutput = embedding->onForward({inputIds, tokenTypeIds});
                // auto ptr = embeddingOutput[0]->readMap<float>();
                auto encoderOutput = encoder->onForward(
                        {embeddingOutput[0], extendedAttentionMask});
                // auto ptr = encoderOutput[0]->readMap<float>();
                auto poolerOutput = pooler->onForward({encoderOutput[0]});
                // auto ptr = poolerOutput[0]->readMap<float>();
                return {encoderOutput[0], poolerOutput[0]};
            }

            BERTForClassification::BERTForClassification(int vocabSize, int hidden, int nLayers,
                                                         int attnHeads,
                                                         int intermediateSize,
                                                         float attentionDropout,
                                                         float hiddenDropout, int numClasses,
                                                         bool forParallel) {
                this->nLayers = nLayers;

                this->bert = std::make_shared<BERT>(vocabSize, hidden, nLayers, attnHeads,
                                                    intermediateSize, attentionDropout,
                                                    forParallel);
                this->dropout.reset(NN::Dropout(hiddenDropout));
                this->classifier.reset(NN::Linear(hidden, numClasses, true));
                registerModel({bert, dropout, classifier});
            }

            std::vector<Express::VARP>
            BERTForClassification::onForward(const std::vector<Express::VARP> &inputs) {
                auto sequenceOutput = bert->onForward(inputs);
                auto output = dropout->onForward(sequenceOutput);
                output = classifier->onForward({output[0]});
                return output;
            }

            SubBERTForClassification::SubBERTForClassification(int start, int end,
                                                               std::map<std::string, double> &args) {
                this->totalLayers = (int) args["total_layer"];
                int numClasses = (int) args["n_class"];
                int hidden = (int) args["hidden_size"];
                int vocabSize = (int) args["vocab_size"];
                this->numHiddenLayers = (int) args["num_hidden_layers"];

                float attentionDropout = (float) args["attention_dropout_prob"];
                float hiddenDropout = (float) args["hidden_dropout_prob"];

                int numAttentionHeads = (int) args["num_attention_heads"];
                int intermediateSize = (int) args["intermediate_size"];

                encoders = std::vector<std::shared_ptr<Express::Module> >();

                if (end == -1) {
                    end = totalLayers - 1;
                }

                if (start == 0) {
                    this->embedding = std::make_shared<BERTEmbedding>(vocabSize, hidden, 512, 2,
                                                                      attentionDropout);
                    registerModel({this->embedding});
                }


                int curLayer = 1;
                for (int i = 0; i < numHiddenLayers; i++) {
                    if (curLayer >= start && curLayer <= end) {
                        encoders.push_back(std::make_shared<BERTLayer>(numAttentionHeads, hidden,
                                                                       intermediateSize,
                                                                       attentionDropout));
                    }
                    curLayer++;
                }

                if (!encoders.empty()) {
                    registerModel(encoders);
                }

                int poolerLayerIdx = numHiddenLayers + 1;
                if (poolerLayerIdx >= start && poolerLayerIdx <= end) {
                    this->pooler = std::make_shared<BERTPooler>(hidden);
                    registerModel({this->pooler});
                }

                if (end == totalLayers - 1) {
                    this->dropout.reset(NN::Dropout(hiddenDropout));
                    this->classifier.reset(NN::Linear(hidden, numClasses, true));
                    registerModel({this->dropout, this->classifier});
                }
            }

            std::vector<Express::VARP>
            SubBERTForClassification::onForward(const std::vector<Express::VARP> &inputs) {
                using namespace Express;
                VARP x = inputs[0];

                if (embedding != nullptr) {
                    VARP tokenTypeIds;

                    if (inputs.size() < 3) {
                        // tokenTypeIds is None
                        std::vector<int> allZeros(x->getInfo()->size, 0);
                        tokenTypeIds = _Const(allZeros.data(), x->getInfo()->dim, NCHW,
                                              halide_type_of<int>());
                    } else {
                        tokenTypeIds = inputs[2];
                    }
                    x = embedding->onForward({x, tokenTypeIds})[0];
                }

                if (!encoders.empty()) {
                    VARP attentionMask;
                    if (inputs.size() < 2) {
                        // attentionMask is None
                        std::vector<int> maskVec;
                        auto inputPtr = x->readMap<int>();
                        std::vector<int> dims = x->getInfo()->dim;
                        for (int i = 0; i < dims[0]; i++) {
                            for (int j = 0; j < dims[1]; j++) {
                                if (inputPtr[i * dims[1] + j] != 0) {
                                    maskVec.push_back(1);
                                } else {
                                    maskVec.push_back(0);
                                }
                            }
                        }
                        attentionMask = _Const(maskVec.data(), x->getInfo()->dim, NCHW,
                                               halide_type_of<float>());
                    } else {
                        attentionMask = inputs[1];
                    }

                    for (const auto &encoder: encoders) {
                        x = encoder->onForward({x, attentionMask})[0];
                    }
                }

//                if (pooler != nullptr) {
//                    x = pooler->onForward({x})[0];
//                }

                if (classifier != nullptr) {
                    x = dropout->onForward({x})[0];
                    x = classifier->onForward({x})[0];
                }

                return {x};
            }

            void SubBERTForClassification::loadParamByLayer(int layer, std::string &weightsBasePath,
                                                            int startLayer) {
                if (layer >= this->totalLayers - 1) {
                    // No weights for classifier
                    return;
                }

                if (startLayer == 0) {
                    // Since the startLayer in Bert refers to the start of the encoder layers, we need to add 1 to the startLayer
                    startLayer = 1;
                }
                std::string path = weightsBasePath + "/bert_" + std::to_string(layer) + ".mnn";
                auto params = Variable::load(path.c_str());

                if (layer == 0) {
                    embedding->loadParameters(params);
                } else if (layer == this->numHiddenLayers + 2) {
                    classifier->loadParameters(params);
                } else if (layer == this->numHiddenLayers + 1) {
                    // pooler
                    for (auto &para: params) {
                        para.fix(VARP::TRAINABLE);
                    }
                    pooler->loadParameters(params);
                } else {
                    for (auto &para: params) {
                        para.fix(VARP::TRAINABLE);
                    }
                    encoders[layer - startLayer]->loadParameters(params);
                }
            }
        }
    }
}