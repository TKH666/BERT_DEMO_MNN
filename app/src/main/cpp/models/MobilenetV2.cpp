//
// Created by 陈宇豪 on 2023/1/5.
//

#include "MobileNetV2.h"
#include "log.h"

namespace MNN {
    using namespace MNN::Express;

    int makeDivisible(int v, int divisor, int minValue = 0) {
        if (minValue == 0) {
            minValue = divisor;
        }
        int newV = std::max(minValue, int(v + divisor / 2) / divisor * divisor);

        // Make sure that round down does not go down by more than 10%.
        if (newV < 0.9 * v) {
            newV += divisor;
        }

        return newV;
    }

    class _ConvBnRelu : public Module {
    public:
        _ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1, bool depthwise = false);

        virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

        std::shared_ptr<Module> conv;
        std::shared_ptr<Module> bn;
    };

    std::vector<Express::VARP> _ConvBnRelu::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        x = conv->forward(x);
        x = bn->forward(x);
        x = _Relu6(x);

        return {x};
    }

    _ConvBnRelu::_ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize, int stride, bool depthwise) {
        int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

        NN::ConvOption convOption;
        convOption.kernelSize = {kernelSize, kernelSize};
        convOption.channel    = {inputChannels, outputChannels};
        convOption.padMode    = Express::SAME;
        convOption.stride     = {stride, stride};
        convOption.depthwise  = depthwise;
        conv.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

        bn.reset(NN::BatchNorm(outputChannels));

        registerModel({conv, bn});
    }

    std::shared_ptr<Module> ConvBnRelu(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1,
                                       bool depthwise = false) {
        return std::shared_ptr<Module>(new _ConvBnRelu(inputOutputChannels, kernelSize, stride, depthwise));
    }

    class _BottleNeck : public Module {
    public:
        _BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio);

        virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

        std::vector<std::shared_ptr<Module> > layers;
        bool useShortcut = false;
    };

    std::shared_ptr<Module> BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio) {
        return std::shared_ptr<Module>(new _BottleNeck(inputOutputChannels, stride, expandRatio));
    }

    _BottleNeck::_BottleNeck(std::vector<int> inputOutputChannels, int stride, int expandRatio) {
        int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];
        int expandChannels = inputChannels * expandRatio;

        if (stride == 1 && inputChannels == outputChannels) {
            useShortcut = true;
        }

        if (expandRatio != 1) {
            layers.emplace_back(ConvBnRelu({inputChannels, expandChannels}, 1));
        }

        layers.emplace_back(ConvBnRelu({expandChannels, expandChannels}, 3, stride, true));

        NN::ConvOption convOption;
        convOption.kernelSize = {1, 1};
        convOption.channel    = {expandChannels, outputChannels};
        convOption.padMode    = Express::SAME;
        convOption.stride     = {1, 1};
        convOption.depthwise  = false;
        layers.emplace_back(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

        layers.emplace_back(NN::BatchNorm(outputChannels));

        registerModel(layers);
    }

    std::vector<Express::VARP> _BottleNeck::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        for (int i = 0; i < layers.size(); i++) {
            x = layers[i]->forward(x);
        }

        if (useShortcut) {
            x = x + inputs[0];
        }

        return {x};
    }

    MobileNetV2::MobileNetV2(int numClasses, float widthMult, int divisor) {
        int inputChannels = 32;
        int lastChannels = 1280;

        std::vector<std::vector<int> > invertedResidualSetting;
        invertedResidualSetting.push_back({1, 16, 1, 1});
        invertedResidualSetting.push_back({6, 24, 2, 2});
        invertedResidualSetting.push_back({6, 32, 3, 2});
        invertedResidualSetting.push_back({6, 64, 4, 2});
        invertedResidualSetting.push_back({6, 96, 3, 1});
        invertedResidualSetting.push_back({6, 160, 3, 2});
        invertedResidualSetting.push_back({6, 320, 1, 1});

        inputChannels = makeDivisible(inputChannels * widthMult, divisor);
        lastChannels = makeDivisible(lastChannels * std::max(1.0f, widthMult), divisor);

        firstConv = ConvBnRelu({3, inputChannels}, 3, 2);

        for (int i = 0; i < invertedResidualSetting.size(); i++) {
            std::vector<int> setting = invertedResidualSetting[i];
            int t                    = setting[0];
            int c                    = setting[1];
            int n                    = setting[2];
            int s                    = setting[3];

            int outputChannels = makeDivisible(c * widthMult, divisor);

            for (int j = 0; j < n; j++) {
                int stride = 1;
                if (j == 0) {
                    stride = s;
                }

                bottleNeckBlocks.emplace_back(BottleNeck({inputChannels, outputChannels}, stride, t));
                inputChannels = outputChannels;
            }
        }

        lastConv = ConvBnRelu({inputChannels, lastChannels}, 1);

        dropout.reset(NN::Dropout(0.1));
        fc.reset(NN::Linear(lastChannels, numClasses, true, std::shared_ptr<Initializer>(Initializer::MSRA())));

        registerModel({firstConv, lastConv, dropout, fc});
        registerModel(bottleNeckBlocks);
    }

    MobileNetV2::~MobileNetV2() {

    }

    std::vector<Express::VARP> MobileNetV2::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        x = firstConv->forward(x);

        for (int i = 0; i < bottleNeckBlocks.size(); i++) {
            x = bottleNeckBlocks[i]->forward(x);
        }

        x = lastConv->forward(x);

        // global avg pooling
        x = _AvePool(x, {-1, -1});

        x = _Convert(x, NCHW);
        x = _Reshape(x, {0, -1});

        x = dropout->forward(x);
        x = fc->forward(x);

        x = _Softmax(x, 1);
        return {x};
    }

    SubMobileNetV2::SubMobileNetV2(int start, int end, std::map<std::string, double>& args) {
        int inputChannels = 32;
        int lastChannels = 1280;

        int totalLayers = (int) args["total_layer"];
        int numClasses = (int) args["n_class"];

        std::vector<std::vector<int> > invertedResidualSetting;
        invertedResidualSetting.push_back({1, 16, 1, 1});
        invertedResidualSetting.push_back({6, 24, 2, 2});
        invertedResidualSetting.push_back({6, 32, 3, 2});
        invertedResidualSetting.push_back({6, 64, 4, 2});
        invertedResidualSetting.push_back({6, 96, 3, 1});
        invertedResidualSetting.push_back({6, 160, 3, 2});
        invertedResidualSetting.push_back({6, 320, 1, 1});

//        features = std::vector<std::shared_ptr<Express::Module> >();
//        classifier = std::vector<std::shared_ptr<Express::Module> >();

        if (end == -1) {
            end = totalLayers - 1;
        }

        if (end == totalLayers - 1) {
            classifier.emplace_back(NN::Linear(lastChannels, numClasses, true, std::shared_ptr<Initializer>(Initializer::MSRA())));
        }

        if (start == 0) {
            features.emplace_back(ConvBnRelu({3, inputChannels}, 3, 2));
        }

        double widthMult = args["width_mult"];
        int divisor = 8;
        int curLayer = 1;

        for (int i = 0; i < invertedResidualSetting.size(); i++) {
            std::vector<int> setting = invertedResidualSetting[i];
            int t                    = setting[0];
            int c                    = setting[1];
            int n                    = setting[2];
            int s                    = setting[3];

            int outputChannels = makeDivisible(c * widthMult, divisor);

            for (int j = 0; j < n; j++) {
                int stride = 1;
                if (curLayer >= start && curLayer <= end) {
                    if (j == 0) {
                        stride = s;
                    }
                    features.emplace_back(BottleNeck({inputChannels, outputChannels}, stride, t));
                }
                inputChannels = outputChannels;
                curLayer++;
                if (curLayer > end) {
                    break;
                }
            }
        }

        if (start < totalLayers - 1 && end >= totalLayers - 2) {
            // The last conv
            features.emplace_back(ConvBnRelu({inputChannels, lastChannels}, 1));
        }

        dropout.reset(NN::Dropout(0.1));

        originFeaturesLen = totalLayers - 1;
        originClassifierLen = 1;

        registerModel({dropout});
        registerModel(features);
        registerModel(classifier);
    }

    std::vector<Express::VARP> SubMobileNetV2::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        for (int i = 0; i < features.size(); i++) {
            x = features[i]->forward(x);
        }

        if (classifier.size() > 0) {
            x = _AvePool(x, {-1, -1});

            x = _Convert(x, NCHW);
            x = _Reshape(x, {0, -1});

            x = dropout->forward(x);

            x = classifier[0]->forward(x);
            x = _Softmax(x, 1);
        }

        return {x};
    }

    void SubMobileNetV2::loadParamByLayer(int layer, std::string& weightsBasePath, int startLayer) {
        // TODO: Load pretrained parameters for the SubMobileNetV2
    }

}