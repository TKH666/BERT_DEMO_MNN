//
// Created by 陈宇豪 on 2023/1/10.
//

#ifndef FTPIPEHD_MNN_DATASETS_H
#define FTPIPEHD_MNN_DATASETS_H

#include <string>
#include <DataLoader.hpp>

namespace FTPipeHD {
    class Datasets {
    public:
        Datasets(std::string basePath, std::string name, std::string path, int trainBatchSize=64, int testBatchSize=8);
        static std::shared_ptr<MNN::Train::DataLoader> trainSetLoader;
        static std::shared_ptr<MNN::Train::DataLoader> testSetLoader;
        static Datasets *datasets;
    };
}

#endif //FTPIPEHD_MNN_DATASETS_H
