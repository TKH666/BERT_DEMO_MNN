//
// Created by 陈宇豪 on 2023/1/10.
//

#ifndef FTPIPEHD_MNN_MODEL_H
#define FTPIPEHD_MNN_MODEL_H

#include <string>
#include "MNN/expr/Module.hpp"
#include "SubModel.h"

using namespace MNN::Express;
namespace FTPipeHD {

    class ModelZoo {
    public:
        static void createModel(std::string& modelName, std::map<std::string, double> &modelArgs);
        static void createSubModel(std::string& modelName, std::map<std::string, double> &modelArgs, int start, int end);
        static std::shared_ptr<Module> modelPtr;
        static std::shared_ptr<SubModel> subModelPtr;
    };
}

#endif //FTPIPEHD_MNN_MODEL_H
