//
// Created by 陈宇豪 on 2023/2/24.
//

#include "commonStates.h"

using namespace MNN;
using namespace MNN::Express;

namespace FTPipeHD {
    int CommonStates::batchSize = 0;
    std::map<int, std::vector<VARP> > CommonStates::intermediatePool{};
    std::map<int, int> CommonStates::idToWeight{};

    void CommonStates::setBatchSize(int _batchSize) {
        batchSize = _batchSize;
    }

    void CommonStates::storeIntermediate(int iterId, VARP data, int type) {
        if (intermediatePool.find(iterId) == intermediatePool.end()) {
            intermediatePool[iterId] = std::vector<VARP>(2);
        }

        intermediatePool[iterId][type] = data;
    }

    VARP CommonStates::getIntermediate(int iterId, int type) {
        return intermediatePool[iterId][type];
    }

    void CommonStates::removeIntermediate(int iterId) {
        intermediatePool.erase(iterId);
    }

    void CommonStates::setWeightVersion(int iterId, int version) {
        idToWeight[iterId] = version;
    }

    int CommonStates::getWeightVersion(int iterId) {
        return idToWeight[iterId];
    }


}