//
// Created by 陈宇豪 on 2023/6/13.
//

#include "DemoUnit.hpp"
#include "NN.hpp"
#include "MEmbedding.hpp"
#include "BERT.cpp"
#include "BERTEmbedding.cpp"
#include "Initializer.hpp"
#include "BERTTokenizer.hpp"
#include "BERTLayer.cpp"
#include <iostream>
#include "Conll2003Dataset.hpp"
#include "BertUtils.hpp"
#include "Loss.hpp"
#include "SGD.hpp"
#include "RandomGenerator.hpp"
#include <MNN/AutoTime.hpp>
#include "LearningRateScheduler.hpp"

using namespace std;
using namespace MNN::Train;
using namespace MNN::Express;
using namespace MNN::Train::Model;

std::vector<VARP> loadLinearParams(std::map<std::string, VARP>& varMap, std::string& weightName) {
    auto var = varMap[weightName];
    auto varExpr = var->expr().first;
    auto varConv = NN::Utils::ExtractConvolution(varExpr);
    return {varConv.weight, varConv.bias};
}

class LoadDatasetTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out LoadDatasetTest /path/to/data/" << endl;
            return 0;
        }
        std::string vocabPath = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/vocab_files/bert-base-cased-vocab.txt";
        std::string root = argv[1];
        auto trainDataset = Conll2003Dataset::create(root, Conll2003Dataset::Mode::TEST);
        int batchSize = 8;
        int numWorkers = 1;
        bool shuffle = false;
        auto dataLoader = std::shared_ptr<DataLoader>(trainDataset.createLoader(batchSize, true, shuffle, numWorkers));

        for (int i = 0 ; i < 3; i++) {
            auto trainData  = dataLoader->next()[0];
            auto input = trainData.first[0];
            auto label = trainData.second[0];
            auto inputPtr = input->readMap<int>();
            auto labelPtr = label->readMap<int>();
            MNN_PRINT("hi");
        }
        MNN_PRINT("LoadDatasetTest Success!\n");
    }
};

class MEmbeddingTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        // input 的维度可以乱序
        std::vector<int> inputShape = {2, 3};
        std::vector<int> vectorInput(inputShape[0] * inputShape[1]);
        vectorInput[0] = 1, vectorInput[1] = 2, vectorInput[2] = 3;
        vectorInput[3] = 3, vectorInput[4] = 2, vectorInput[5] = 5;
        auto input = _Const(vectorInput.data(), inputShape, NCHW, halide_type_of<int>());

        auto embedding = MEmbedding(10, 3);
        auto output = embedding.forward(input);

        auto outputPtr = output->readMap<float>();
        auto outputSize = output->getInfo()->size;
        MNN_PRINT("Selected index: ");
        for (int i = 0; i < outputSize; i++) {
            MNN_PRINT("%f ", outputPtr[i]);
        }
    }
};

class PositionalEmbeddingTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        std::vector<int> inputShape = {1, 3, 4};
        std::vector<float> allOneInput(inputShape[0] * inputShape[1] * inputShape[2], 1.0);
        auto input    = _Input(inputShape, NCHW);
        auto inputPtr = input->writeMap<float>();
        ::memcpy(inputPtr, allOneInput.data(), allOneInput.size() * sizeof(float));

        auto pe = PositionalEmbedding(10, 10);
        auto output = pe->forward(input);

        auto outputPtr = output->readMap<float>();
        auto outputSize = output->getInfo()->size;
        MNN_PRINT("Positional embedding: ");

        for (int i = 0; i < outputSize; i++) {
            MNN_PRINT("%f ", outputPtr[i]);
        }
    }
};

class LoadTorchBERTTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        std::string path = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/MNN_Converter/build/mnn_models/bert_embedding.mnn";
        auto vars = Variable::load(path.c_str());
        std::set<std::string> nameSet = {"word_embeddings.weight", "position_embeddings.weight", "token_type_embeddings.weight", "LayerNorm.weight", "LayerNorm.bias"};
        std::map<std::string, VARP> varMap;

        for (auto var : vars) {
            if (nameSet.find(var->name()) != nameSet.end()) {
                varMap[var->name()] = var;
            }
        }

        // Load bert embedding
        auto bertEmbedding = BERTEmbedding(30522, 768, 512, 2);
        bertEmbedding.wordEmbeddings->loadParameters({varMap["word_embeddings.weight"]});
        bertEmbedding.positionEmbeddings->loadParameters({varMap["position_embeddings.weight"]});
        bertEmbedding.tokenTypeEmbeddings->loadParameters({varMap["token_type_embeddings.weight"]});
        bertEmbedding.layerNorm->loadParameters({varMap["LayerNorm.weight"], varMap["LayerNorm.bias"]});

        std::string savePath = "bertEmbedding.snapshot.mnn";
        Variable::save(bertEmbedding.parameters(), savePath.c_str());

        MNN_PRINT("hi");
    }
};

class TokenizerTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        // std::string path = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/MNN_Converter/build/mnn_models/bert_embedding.mnn";
        std::string path = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/vocab_files/bert-base-uncased-vocab.txt";
        // auto vocab = loadVocab("");

        // std::string input = "Hello, my dog is cute";
        std::string input = "I'm doing great state-of-the-art, thank you!";
        auto bertTokenizer = BERTTokenizer(path);
        auto tokens = bertTokenizer.tokenize(input);
        auto ids = bertTokenizer.convertTokensToIds(tokens.first);
        auto ptr = ids->readMap<int>();
        MNN_PRINT("hi");
    }
};

class BERTEncoderTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        // std::string weight_path = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/MNN_Converter/build/mnn_models/bert_encoder_layer_0.mnn";
        // std::string weight_path = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/MNN_Converter/build/mnn_models/bert_encoder_layer_0_attention_self_query.mnn";
        std::string weight_path = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/MNN_Converter/build/mnn_models/bert_encoder.mnn";

        // Load pre-trained weights of encoder
        auto varMap = Variable::loadMap(weight_path.c_str());

        std::set<std::string> attentionSet = {"/attention/self/query/Add_output_0__matmul_converted", "/attention/self/key/Add_output_0__matmul_converted", "/attention/self/value/Add_output_0__matmul_converted",
                                              "/attention/output/dense/Add_output_0__matmul_converted"};

        int numHiddenLayers = 12;
        int hiddenSize = 768;
        int intermediateSize = 3072;
        int numAttentionHeads = 12;
        // auto bertLayer = std::make_shared<BERTLayer>(numAttentionHeads, hiddenSize, intermediateSize, 0.1);
        auto bertEncoder = std::make_shared<BERTEncoder>(numHiddenLayers, numAttentionHeads, hiddenSize, intermediateSize, 0.1);

        for (int l = 0; l < numHiddenLayers; l++) {
            std::shared_ptr<BERTLayer> layer = std::dynamic_pointer_cast<BERTLayer>(bertEncoder->layer[l]);
            for (auto& attentionName : attentionSet) {
                std::string curName = "/layer." + std::to_string(l) + attentionName;
                auto var = varMap[curName];
                auto varExpr = var->expr().first;
                auto varConv = NN::Utils::ExtractConvolution(varExpr);
                auto varWeight = varConv.weight;
                auto varBias = varConv.bias;

                // here the bias should be cloned, otherwise the _Unsqueeze will be treated as an operator
                varBias = _Unsqueeze(varBias, {0});
                varWeight = _Squeeze(varWeight, {2, 3});

                auto varBias_ = _Const(varBias->readMap<float>(), varBias->getInfo()->dim, NCHW);
                auto varWeight_ = _Const(varWeight->readMap<float>(), varWeight->getInfo()->dim, NCHW);

                if (attentionName == "/attention/self/query/Add_output_0__matmul_converted") {
                    layer->attention->self->query->loadParameters({varBias_, varWeight_});
                } else if (attentionName == "/attention/self/key/Add_output_0__matmul_converted") {
                    layer->attention->self->key->loadParameters({varBias_, varWeight_});
                } else if (attentionName == "/attention/self/value/Add_output_0__matmul_converted") {
                    layer->attention->self->value->loadParameters({varBias_, varWeight_});
                } else if (attentionName == "/attention/output/dense/Add_output_0__matmul_converted") {
                    layer->attention->output->dense->loadParameters({varBias_, varWeight_});
                }
            }

            layer->attention->output->layerNorm->loadParameters({varMap["layer." + std::to_string(l) + ".attention.output.LayerNorm.weight"], varMap["layer." + std::to_string(l) + ".attention.output.LayerNorm.bias"]});

            // Load intermediate
            auto var = varMap["/layer." + std::to_string(l) + "/intermediate/dense/Add_output_0__matmul_converted"];
            auto varExpr = var->expr().first;
            auto varConv = NN::Utils::ExtractConvolution(varExpr);
            auto varWeight = varConv.weight;
            auto varBias = varConv.bias;
            varBias = _Unsqueeze(varBias, {0});
            varWeight = _Squeeze(varWeight, {2, 3});

            auto varBias_ = _Const(varBias->readMap<float>(), varBias->getInfo()->dim, NCHW);
            // varBias_.fix(VARP::TRAINABLE);
            auto varWeight_ = _Const(varWeight->readMap<float>(), varWeight->getInfo()->dim, NCHW);
            // varWeight_.fix(VARP::TRAINABLE);

            layer->intermediate->dense->loadParameters({varBias_, varWeight_});

            // Load output
            var = varMap["/layer." + std::to_string(l) + "/output/dense/Add_output_0__matmul_converted"];
            varExpr = var->expr().first;
            varConv = NN::Utils::ExtractConvolution(varExpr);
            varWeight = varConv.weight;
            varBias = varConv.bias;
            varBias = _Unsqueeze(varBias, {0});
            varWeight = _Squeeze(varWeight, {2, 3});

            varBias_ = _Const(varBias->readMap<float>(), varBias->getInfo()->dim, NCHW);
            varWeight_ = _Const(varWeight->readMap<float>(), varWeight->getInfo()->dim, NCHW);
            // auto temp = varBias_->readMap<float>();
            layer->output->dense->loadParameters({varBias_, varWeight_});

            layer->output->layerNorm->loadParameters({varMap["layer." + std::to_string(l) + ".output.LayerNorm.weight"], varMap["layer." + std::to_string(l) + ".output.LayerNorm.bias"]});
        }

        // 192 params
//        std::string savePath = "bertEncoder.snapshot.mnn";
//        Variable::save(bertEncoder->parameters(), savePath.c_str());

        // save the model by layer
        for (int l = 0; l < numHiddenLayers; l++) {
            std::string savePath = "bert_" + std::to_string(l + 1) + ".mnn";
            Variable::save(bertEncoder->layer[l]->parameters(), savePath.c_str());
        }
        MNN_PRINT("BERTEncoder Test Passed!");
    }
};

class BERTPoolerTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        std::string weight_path = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/MNN_Converter/build/mnn_models/bert_pooler.mnn";
        auto varMap = Variable::loadMap(weight_path.c_str());
        int hiddenSize = 768;
        auto bertPooler = BERTPooler(768);

        auto var = varMap["/dense/Gemm_output_0__matmul_converted"];
        auto varExpr = var->expr().first;
        auto varConv = NN::Utils::ExtractConvolution(varExpr);
        auto varWeight = varConv.weight;
        auto varBias = varConv.bias;
        varBias = _Unsqueeze(varBias, {0});
        varWeight = _Squeeze(varWeight, {2, 3});

        auto varBias_ = _Const(varBias->readMap<float>(), varBias->getInfo()->dim, NCHW);
        auto varWeight_ = _Const(varWeight->readMap<float>(), varWeight->getInfo()->dim, NCHW);

        bertPooler.loadParameters({varBias_, varWeight_});
        std::string savePath = "bertPooler.snapshot.mnn";
        Variable::save(bertPooler.parameters(), savePath.c_str());
        MNN_PRINT("BERTPooler Test Passed!");
    }
};

class BERTTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        std::string vocab_path = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/vocab_files/bert-base-uncased-vocab.txt";

        // TODO: Loaded params is CONST now, not TRAINABLE
        auto bertEmbedding = BERTEmbedding(30522, 768, 512, 2, 0.0);
        auto bertEmbeddingParams = Variable::load("/home/ubuntu/yyx/MNN-LLM/pretrained_weights/bert/bertEmbedding.snapshot.mnn");
        bertEmbedding.loadParameters(bertEmbeddingParams);

        // Load pre-trained weights of encoder
        int numHiddenLayers = 12;
        int hiddenSize = 768;
        int intermediateSize = 3072;
        int numAttentionHeads = 12;
        auto bertEncoder = BERTEncoder(numHiddenLayers, numAttentionHeads, hiddenSize, intermediateSize, 0.0, true);
        auto bertEncoderParams = Variable::load("/home/ubuntu/yyx/MNN-LLM/pretrained_weights/bert/bertEncoder.snapshot.mnn");
        bertEncoder.loadParameters(bertEncoderParams);

        // Load pre-trained weights of pooler
        auto bertPooler = BERTPooler(hiddenSize);
        auto bertPoolerParams = Variable::load("/home/ubuntu/yyx/MNN-LLM/pretrained_weights/bert/bertPooler.snapshot.mnn");
        bertPooler.loadParameters(bertPoolerParams);

        // Input test
        std::string input = "How are you?";
        // std::string input = "I'm doing great state-of-the-art, thank you!";
        auto bertTokenizer = BERTTokenizer(vocab_path);
        auto tokens = bertTokenizer.tokenize(input);
        auto ids = bertTokenizer.convertTokensToIds(tokens.first);

        std::vector<int> allZeros(ids->getInfo()->size, 0);
        auto tokenTypeIds = _Const(allZeros.data(), ids->getInfo()->dim, NCHW, halide_type_of<int>());

        std::vector<float> allOnes(ids->getInfo()->size, 1.0);
        auto attentionMask = _Const(allOnes.data(), ids->getInfo()->dim, NCHW, halide_type_of<float>());

        auto extendedAttentionMask = _Unsqueeze(attentionMask, {1, 2});
        extendedAttentionMask = (_Scalar<float>(1.0) - extendedAttentionMask) * _Scalar<float>(-10000.0);

        auto embeddingOutput = bertEmbedding.onForward({ids, tokenTypeIds});

        auto encoderOutput = bertEncoder.onForward({embeddingOutput[0], extendedAttentionMask});

        // auto ptr = encoderOutput[0]->readMap<float>();

        auto poolerOutput = bertPooler.onForward({encoderOutput[0]});

        auto ptr = poolerOutput[0]->readMap<float>();
//        for (int i = 0; i < poolerOutput[0]->getInfo()->size; i++) {
//            MNN_PRINT("%f ", ptr[i]);
//        }
        MNN_PRINT("hi");
    }
};

class BERTTrainTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out BERTTrainTest /path/to/data/" << endl;
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
        std::shared_ptr<BERTForClassification> model(new BERTForClassification(vocabSize, hiddenSize, numHiddenLayers, numAttentionHeads, intermediateSize, attnDropout, hiddenDropout, 9, false));

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
        MNN_PRINT("hi");
    }
};

class BERTPartitioningTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out BERTPartitioningTest /path/to/data/" << endl;
            return 0;
        }
        std::string vocabPath = "/Users/fubuki/Desktop/NESC/Edge Large Model Training/data/vocab_files/bert-base-uncased-vocab.txt";
        std::string root = argv[1];

        std::map<std::string, double> args;
        args["total_layer"] = 15.0;
        args["n_class"] = 9.0;
        args["hidden_size"] = 768.0;
        args["vocab_size"] = 30522.0;
        args["num_hidden_layers"] = 12.0;

        args["attention_dropout_prob"] = 0.0;
        args["hidden_dropout_prob"] = 0.0;

        args["num_attention_heads"] = 12.0;
        args["intermediate_size"] = 3072.0;

        std::shared_ptr<SubBERTForClassification> submodel1(new SubBERTForClassification(0, 5, args));
        for (int i = 0; i <= 5; i++) {
            submodel1->loadParamByLayer(i);
        }

        std::shared_ptr<SubBERTForClassification> submodel2(new SubBERTForClassification(6, 12, args));
        for (int i = 6; i <= 12; i++) {
            submodel2->loadParamByLayer(i, 6);
        }

        std::shared_ptr<SubBERTForClassification> submodel3(new SubBERTForClassification(13, -1, args));
        for (int i = 13; i <= 14; i++) {
            submodel3->loadParamByLayer(i, 13);
        }

        std::shared_ptr<SGD> sgd1(new SGD(submodel1));
        sgd1->setMomentum(0.9f);
        sgd1->setWeightDecay(0.0005f);

        std::shared_ptr<SGD> sgd2(new SGD(submodel2));
        sgd2->setMomentum(0.9f);
        sgd2->setWeightDecay(0.0005f);

        std::shared_ptr<SGD> sgd3(new SGD(submodel3));
        sgd3->setMomentum(0.9f);
        sgd3->setWeightDecay(0.0005f);

        auto exe = Executor::getGlobalExecutor();
        MNN::BackendConfig config;
        exe->setGlobalExecutorConfig(MNN_FORWARD_CUDA, config, 4);

        auto dataset = Conll2003Dataset::create(root, Conll2003Dataset::Mode::TRAIN);
        // the stack transform, stack [1, 28, 28] to [n, 1, 28, 28]
        const size_t batchSize  = 1;
        const size_t numWorkers = 0;
        bool shuffle            = true;

        auto dataLoader = std::shared_ptr<DataLoader>(dataset.createLoader(batchSize, true, shuffle, numWorkers));

        submodel1->clearCache();
        submodel2->clearCache();
        submodel3->clearCache();

        exe->gc(Executor::FULL);
        exe->resetProfile();

        dataLoader->reset();
        submodel1->setIsTraining(true);
        submodel2->setIsTraining(true);
        submodel3->setIsTraining(true);

        auto trainData  = dataLoader->next();
        auto example    = trainData[0];
        auto label = example.second[0];

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
        auto extendedAttentionMask = _Unsqueeze(attentionMask, {1, 2});
        extendedAttentionMask = (_Scalar<float>(1.0) - extendedAttentionMask) * _Scalar<float>(-10000.0);

        auto output1 = submodel1->onForward({example.first[0], extendedAttentionMask})[0];
        auto output1Ptr = output1->readMap<void>();
        auto output1Info = output1->getInfo();
        auto output1New = _Input(output1Info->dim, output1Info->order, output1Info->type);
        auto output1NewPtr = output1New->writeMap<void>();
        ::memcpy(output1NewPtr, output1Ptr, output1Info->size * output1Info->type.bytes());

        auto extendedAttentionMaskInfo = extendedAttentionMask->getInfo();
        auto extendedAttentionMaskPtr = extendedAttentionMask->readMap<void>();
        auto extendedAttentionMaskNew = _Input(extendedAttentionMaskInfo->dim, extendedAttentionMaskInfo->order, extendedAttentionMaskInfo->type);
        auto extendedAttentionMaskNewPtr = extendedAttentionMaskNew->writeMap<void>();
        ::memcpy(extendedAttentionMaskNewPtr, extendedAttentionMaskPtr, extendedAttentionMaskInfo->size * extendedAttentionMaskInfo->type.bytes());

        auto output2 = submodel2->onForward({output1New, extendedAttentionMaskNew})[0];
        auto output2Ptr = output2->readMap<void>();
        auto output2Info = output2->getInfo();
        auto output2New = _Input(output2Info->dim, output2Info->order, output2Info->type);
        auto output2NewPtr = output2New->writeMap<void>();
        ::memcpy(output2NewPtr, output2Ptr, output2Info->size * output2Info->type.bytes());

        auto extendedAttentionMaskInfo2 = extendedAttentionMask->getInfo();
        auto extendedAttentionMaskPtr2 = extendedAttentionMask->readMap<void>();
        auto extendedAttentionMaskNew2 = _Input(extendedAttentionMaskInfo->dim, extendedAttentionMaskInfo->order, extendedAttentionMaskInfo->type);
        auto extendedAttentionMaskNewPtr2 = extendedAttentionMaskNew2->writeMap<void>();
        ::memcpy(extendedAttentionMaskNewPtr2, extendedAttentionMaskPtr2, extendedAttentionMaskInfo->size * extendedAttentionMaskInfo->type.bytes());

        auto output3 = submodel3->onForward({output2New, extendedAttentionMaskNew2})[0];

        auto outputPtr = output3->readMap<float>();

        // Calculate the loss
        // TODO: Find the active loss, which can be optimized
        auto activeLoss = _Const(maskVecIdx.data(), {(int)maskVecIdx.size()}, NCHW, halide_type_of<int>());
        auto activeLogits = _GatherV2(_Reshape(output3, {-1, output3->getInfo()->dim[2]}), activeLoss, _Scalar<int>(0));
        auto activeLabels = _GatherV2(_Reshape(label, {-1}), activeLoss, _Scalar<int>(0));
        auto newActiveLabels = _OneHot(_Cast<int32_t>(activeLabels), _Scalar<int>(9), _Scalar<float>(1.0f), _Scalar<float>(0.0f));

        int ignoreIndex = -100;
        auto ignoredMask = _Cast<float>(_Unsqueeze(_NotEqual(activeLabels, _Scalar<int>(ignoreIndex)), {1}));
        newActiveLabels = newActiveLabels * ignoredMask;

//        auto labelsPtr = newActiveLabels->readMap<float>();
//        auto logitsPtr = activeLogits->readMap<float>();
        auto loss    = _CrossEntropy(_Softmax(activeLogits), newActiveLabels);
        auto lossPtr = loss->readMap<float>();
        // Variable::save({loss}, "loss.mnn");
        // Backwarding
        std::map<VARP, VARP> grad3;
        auto shape = loss->getInfo();
        auto init= _Const(1.0f, shape->dim, shape->order);

        auto output2Grad = sgd3->backward(loss, output2New, init, grad3);
        auto output2GradPtr = output2Grad[0]->readMap<void>();
        auto output2GradInfo = output2Grad[0]->getInfo();
        auto output2GradNew = _Input(output2GradInfo->dim, output2GradInfo->order, output2GradInfo->type);
        auto output2GradNewPtr = output2GradNew->writeMap<void>();
        ::memcpy(output2GradNewPtr, output2GradPtr, output2GradInfo->size * output2GradInfo->type.bytes());

        sgd3->stepNew(grad3);

        std::map<VARP, VARP> grad2;
        auto output1Grad = sgd2->backward(output2, output1New, output2GradNew, grad2);
        auto output1GradPtr = output1Grad[0]->readMap<void>();
        auto output1GradInfo = output1Grad[0]->getInfo();
        auto output1GradNew = _Input(output1GradInfo->dim, output1GradInfo->order, output1GradInfo->type);
        auto output1GradNewPtr = output1GradNew->writeMap<void>();
        ::memcpy(output1GradNewPtr, output1GradPtr, output1GradInfo->size * output1GradInfo->type.bytes());

        sgd2->stepNew(grad2);

        std::map<VARP, VARP> grad1;
        auto outputGrad = sgd1->backward(output1, example.first[0], output1GradNew, grad2);
//        auto outputGradPtr = output1Grad[0]->readMap<void>();
//        auto outputGradInfo = output1Grad[0]->getInfo();
//        auto outputGradNew = _Input(output1GradInfo->dim, output1GradInfo->order, output1GradInfo->type);
//        auto outputGradNewPtr = output1GradNew->writeMap<void>();
//        ::memcpy(outputGradNewPtr, outputGradPtr, outputGradInfo->size * output1GradInfo->type.bytes());

        sgd1->stepNew(grad2);
        MNN_PRINT("BERTPartitioningTest Success!\n");
    }
};

class LinearSchedulerTest : public DemoUnit {
public:
    virtual int run(int argc, const char* argv[]) override {
        if (argc != 2) {
            cout << "usage: ./runTrainDemo.out LinearSchedulerTest /path/to/data/" << endl;
            return 0;
        }

        int iteration = 1700;
        int totalEpochs = 1;
        int numTrainingSteps = iteration * totalEpochs;
        float rate = 2e-5f;
        for (int i = 0; i < iteration; i++) {
            rate = LrScheduler::linear(rate, totalEpochs, numTrainingSteps);
            MNN_PRINT("%d: %.8f\n", i, rate);
        }

        MNN_PRINT("LinearSchedulerTest Success!\n");
    }
};

DemoUnitSetRegister(LoadDatasetTest, "LoadDatasetTest");
DemoUnitSetRegister(MEmbeddingTest, "MEmbeddingTest");
DemoUnitSetRegister(PositionalEmbeddingTest, "PositionalEmbeddingTest");
DemoUnitSetRegister(LoadTorchBERTTest, "LoadTorchBERTTest");
DemoUnitSetRegister(TokenizerTest, "TokenizerTest");
DemoUnitSetRegister(BERTTest, "BERTTest");
DemoUnitSetRegister(BERTEncoderTest, "BERTEncoderTest");
DemoUnitSetRegister(BERTPoolerTest, "BERTPoolerTest");
DemoUnitSetRegister(BERTTrainTest, "BERTTrainTest");
DemoUnitSetRegister(BERTPartitioningTest, "BERTPartitioningTest");
DemoUnitSetRegister(LinearSchedulerTest, "LinearSchedulerTest");
