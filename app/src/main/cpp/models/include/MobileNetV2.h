//
// Created by 陈宇豪 on 2023/1/5.
//

#ifndef FTPIPEHD_MNN_MOBILENETV2_H
#define FTPIPEHD_MNN_MOBILENETV2_H

#include <vector>
#include "MNN/expr/Module.hpp"
#include "nn/Initializer.hpp"
#include "nn/NN.hpp"
#include <map>
#include "SubModel.h"

namespace MNN {
    class MobileNetV2 : public Express::Module {
    public:
        MobileNetV2(int numClasses = 1001, float widthMult = 1.0f, int divisor = 8);

        virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
        ~MobileNetV2();

        std::shared_ptr<Express::Module> firstConv;
        std::vector<std::shared_ptr<Express::Module> > bottleNeckBlocks;
        std::shared_ptr<Express::Module> lastConv;
        std::shared_ptr<Express::Module> dropout;
        std::shared_ptr<Express::Module> fc;
    };

    class SubMobileNetV2 : public FTPipeHD::SubModel {
    public:
        SubMobileNetV2(int start, int end, std::map<std::string, double>& args);

        virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
        virtual void loadParamByLayer(int layer, std::string& weightsBasePath, int startLayer = 1) override;

        std::vector<std::shared_ptr<Express::Module> > features;
        std::vector<std::shared_ptr<Express::Module> > classifier;
        std::shared_ptr<Express::Module> dropout;
        int originFeaturesLen;
        int originClassifierLen = 1;
    };
}
#endif //FTPIPEHD_MNN_MOBILENETV2_H
