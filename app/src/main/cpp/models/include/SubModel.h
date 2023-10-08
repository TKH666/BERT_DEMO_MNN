//
// Created by 陈宇豪 on 2023/7/12.
//

#ifndef FTPIPEHD_MNN_SUBMODEL_H
#define FTPIPEHD_MNN_SUBMODEL_H

#include <string>
#include "MNN/expr/Module.hpp"

using namespace MNN::Express;
namespace FTPipeHD {
    class SubModel : public Module {
    public:
        SubModel() = default;
        virtual ~SubModel() = default;

        virtual void loadParamByLayer(int layer, std::string& weightsBasePath, int startLayer = 0) = 0;
    };
}

#endif //FTPIPEHD_MNN_SUBMODEL_H
