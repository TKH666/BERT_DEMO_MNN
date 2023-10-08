//
// Created by 陈宇豪 on 2023/6/5.
//

#ifndef FTPIPEHD_MNN_SINGLETRAIN_H
#define FTPIPEHD_MNN_SINGLETRAIN_H

#include "MNN/expr/Module.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace FTPipeHD {
    void initTrain(int batchSize);
    void singleTrainOneEpoch(int epoch);
}

#endif //FTPIPEHD_MNN_SINGLETRAIN_H
