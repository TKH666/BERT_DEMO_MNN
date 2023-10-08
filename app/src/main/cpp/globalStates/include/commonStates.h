//
// Created by 陈宇豪 on 2023/2/24.
//

#ifndef FTPIPEHD_MNN_COMMONSTATES_H
#define FTPIPEHD_MNN_COMMONSTATES_H

#include "map"
#include "MNN/expr/Module.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace FTPipeHD {
    class CommonStates {
    private:
        static int batchSize;
        static std::map<int, std::vector<VARP> > intermediatePool;
        static std::map<int, int> idToWeight;
    public:
        static void setBatchSize(int _batchSize);
        static void storeIntermediate(int iterId, VARP data, int type);
        static void removeIntermediate(int iterId);
        static VARP getIntermediate(int iterId, int type);

        static void setWeightVersion(int iterId, int version);
        static int getWeightVersion(int iterId);
    };
}



#endif //FTPIPEHD_MNN_COMMONSTATES_H
