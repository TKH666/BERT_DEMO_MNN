//
// Created by 陈宇豪 on 2023/6/5.
//
#include "MobileViT.h"
#include "MTransformer.h"

namespace MNN {

    using namespace MNN::Express;

    // ConvBnSiLU Definition
    class _ConvBnSiLU : public Module {
    public:
        _ConvBnSiLU(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1,
                    bool depthwise = false);

        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;

        std::shared_ptr<Module> conv;
        std::shared_ptr<Module> bn;
    };

    std::shared_ptr<Module>
    ConvBnSiLU(std::vector<int> inputOutputChannels, int kernelSize = 3, int stride = 1,
               bool depthwise = false) {
        // 提供了抽象层级，提供一致的接口，也可以实现额外的逻辑
        return std::shared_ptr<Module>(
                new _ConvBnSiLU(inputOutputChannels, kernelSize, stride, depthwise));
    }

    _ConvBnSiLU::_ConvBnSiLU(std::vector<int> inputOutputChannels, int kernelSize, int stride,
                             bool depthwise) {
        int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

        NN::ConvOption convOption;
        convOption.kernelSize = {kernelSize, kernelSize};
        convOption.channel = {inputChannels, outputChannels};

        if (kernelSize == 1) {
            // No padding
            convOption.padMode = Express::VALID;
        } else {
            // Padding, corresponding to padding=1 in Pytorch
            convOption.padMode = Express::SAME;
        }

        convOption.stride = {stride, stride};
        // convOption.depthwise  = depthwise;
        conv.reset(NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

        bn.reset(NN::BatchNorm(outputChannels));

        registerModel({conv, bn});
    }

    std::vector<Express::VARP> _ConvBnSiLU::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        x = conv->forward(x);
        x = bn->forward(x);

        // SiLU Implementation
        x = x * _Sigmoid(x);

        return {x};
    }

    class _MV2Block : public Module {
    private:
        bool useResConnect;
    public:
        _MV2Block(std::vector<int> inputOutputChannels, int stride = 1, int expansion = 4);

        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;

        std::vector<std::shared_ptr<Express::Module> > conv;
    };

    std::shared_ptr<Module>
    MV2Block(std::vector<int> inputOutputChannels, int stride = 1, int expansion = 4) {
        return std::shared_ptr<Module>(new _MV2Block(inputOutputChannels, stride, expansion));
    }

    _MV2Block::_MV2Block(std::vector<int> inputOutputChannels, int stride, int expansion) {
        int inputChannels = inputOutputChannels[0], outputChannels = inputOutputChannels[1];

        int hiddenDim = inputChannels * expansion;
        useResConnect = stride == 1 && inputChannels == outputChannels;

        if (expansion != 1) {
            // piece-wise
            conv.emplace_back(ConvBnSiLU({inputChannels, hiddenDim}, 1));
        }

        // depth-wise
        conv.emplace_back(ConvBnSiLU({hiddenDim, hiddenDim}, 3, stride, true));

        // piece-wise linear
        NN::ConvOption convOption;
        convOption.kernelSize = {1, 1};
        convOption.channel = {hiddenDim, outputChannels};
        convOption.padMode = Express::SAME;
        convOption.stride = {1, 1};
        convOption.depthwise = false;
        conv.emplace_back(
                NN::Conv(convOption, false, std::shared_ptr<Initializer>(Initializer::MSRA())));

        conv.emplace_back(NN::BatchNorm(outputChannels));

        registerModel(conv);
    }

    std::vector<Express::VARP> _MV2Block::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        for (int i = 0; i < conv.size(); i++) {
            x = conv[i]->forward(x);
        }

        if (useResConnect) {
            x = x + inputs[0];
        }

        return {x};
    }

    // ConvBnSiLU Definition
    class _MobileViTBlock : public Module {
    private:
        int patchHeight, patchWidth;
    public:
        _MobileViTBlock(int dim, int depth, int channel, int kernelSize,
                        std::vector<int> &patchSize, int mlpDim, float dropout = 0.0);

        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;

        std::shared_ptr<Module> conv1;
        std::shared_ptr<Module> conv2;

        std::shared_ptr<Module> transformer;
        std::shared_ptr<Module> conv3;
        std::shared_ptr<Module> conv4;
    };

    std::shared_ptr<Module>
    MobileViTBlock(int dim, int depth, int channel, int kernelSize, std::vector<int> &patchSize,
                   int mlpDim, float dropout = 0.0) {
        // 提供了抽象层级，提供一致的接口，也可以实现额外的逻辑
        return std::shared_ptr<Module>(
                new _MobileViTBlock(dim, depth, channel, kernelSize, patchSize,
                                    mlpDim, dropout = 0.0));
    }

    _MobileViTBlock::_MobileViTBlock(int dim, int depth, int channel, int kernelSize,
                                     std::vector<int> &patchSize, int mlpDim,
                                     float dropout) {
        patchHeight = patchSize[0], patchWidth = patchSize[1];
        conv1 = ConvBnSiLU({channel, channel}, kernelSize);
        conv2 = ConvBnSiLU({channel, dim}, 1);

        transformer.reset(new MTransformer(dim, depth, 4, 8, mlpDim, dropout));

        conv3 = ConvBnSiLU({dim, channel}, 1);
        conv4 = ConvBnSiLU({2 * channel, channel}, kernelSize);
    }

    std::vector<Express::VARP>
    _MobileViTBlock::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];
        VARP y = _Clone(x, true);

        // Local representation
        x = conv1->forward(x);
        x = conv2->forward(x);

        // Global representation
        auto shape = x->getInfo()->dim;
        x = _Reshape(x, {shape[0], patchWidth * patchHeight,
                         shape[2] * shape[3] / (patchWidth * patchHeight), shape[1]}, NCHW);

        x = transformer->forward(x);
        x = _Reshape(x, shape, NCHW);

        // Fusion
        x = conv3->forward(x);
        x = _Concat({x, y}, 1);
        x = conv4->forward(x);

        return {x};
    }

    MobileViT::MobileViT(std::pair<int, int> imageSize, std::vector<int> dims,
                         std::vector<int> &channels, int numClasses, int expansion, int kernelSize,
                         std::vector<int> patchSize) {
        std::vector<int> L = {2, 4, 3};
        this->imageSize = imageSize;

        conv1 = ConvBnSiLU({3, channels[0]}, 3, 2);

        mv2.emplace_back(MV2Block({channels[0], channels[1]}, 1, expansion));
        mv2.emplace_back(MV2Block({channels[1], channels[2]}, 2, expansion));
        mv2.emplace_back(MV2Block({channels[2], channels[3]}, 1, expansion));
        mv2.emplace_back(MV2Block({channels[2], channels[3]}, 1, expansion)); // repeat
        mv2.emplace_back(MV2Block({channels[3], channels[4]}, 2, expansion));
        mv2.emplace_back(MV2Block({channels[5], channels[6]}, 2, expansion));
        mv2.emplace_back(MV2Block({channels[7], channels[8]}, 2, expansion));

        mvit.emplace_back(MobileViTBlock(dims[0], L[0], channels[5], kernelSize, patchSize,
                                         int(dims[0] * 2)));
        mvit.emplace_back(MobileViTBlock(dims[1], L[1], channels[7], kernelSize, patchSize,
                                         int(dims[1] * 4)));
        mvit.emplace_back(MobileViTBlock(dims[2], L[2], channels[9], kernelSize, patchSize,
                                         int(dims[2] * 4)));

        conv2 = ConvBnSiLU({channels[channels.size() - 2], channels.back()}, 1);

        fc.reset(NN::Linear(channels.back(), numClasses, false));
    }

    std::vector<Express::VARP> MobileViT::onForward(const std::vector<Express::VARP> &inputs) {
        using namespace Express;
        VARP x = inputs[0];

        x = conv1->forward(x);
        x = mv2[0]->forward(x);

        x = mv2[1]->forward(x);
        x = mv2[2]->forward(x);
        x = mv2[3]->forward(x);

        x = mv2[4]->forward(x);
        x = mvit[0]->forward(x);

        x = mv2[5]->forward(x);

        x = mvit[1]->forward(x);

        x = mv2[6]->forward(x);
        x = mvit[2]->forward(x);
        x = conv2->forward(x);

        x = _AvePool(x, {imageSize.first / 32, imageSize.first / 32}, {1, 1}); // ih / 32
        auto shape = x->getInfo()->dim;
        x = _Convert(x, NCHW); // after pooling, the order of x is NC4HW4
        x = _Reshape(x, {-1, shape[1]});
        x = fc->forward(x);

        return {x};
    }
}
