//
// Created by 陈宇豪 on 2023/1/10.
//

#include "datasets.h"
#include "MnistDataset.hpp"
#include "Conll2003Dataset.h"
#include "LambdaTransform.hpp"

namespace FTPipeHD {
    using namespace MNN::Train;
    std::shared_ptr<DataLoader> Datasets::trainSetLoader = nullptr;
    std::shared_ptr<DataLoader> Datasets::testSetLoader = nullptr;
    Datasets *Datasets::datasets = nullptr;

    Example func(Example example) {
        // an easier way to do this
        auto cast       = _Cast(example.first[0], halide_type_of<float>());
        example.first[0] = _Multiply(cast, _Const(1.0f / 255.0f));
        return example;
    }

    Datasets::Datasets(std::string basePath, std::string name, std::string path, int trainBatchSize, int testBatchSize) {
        DatasetPtr trainDataset, testDataset;
        const int trainNumWorkers = 0;
        const int testNumWorkers = 0;
        std::string fullPath = basePath + name + path;

        if (name == "MNIST") {
            trainDataset = MnistDataset::create(fullPath, MnistDataset::Mode::TRAIN);
            testDataset = MnistDataset::create(fullPath, MnistDataset::Mode::TEST);
        } else if (name == "conll2003") {
            trainDataset = Conll2003Dataset::create(basePath, fullPath, Conll2003Dataset::Mode::TRAIN);
            testDataset = Conll2003Dataset::create(basePath, fullPath, Conll2003Dataset::Mode::TEST);
        } else {
            MNN_PRINT("Unknown dataset name: %s\n", name.c_str());
            MNN_ASSERT(false);
        }

        // auto trainTransform = std::make_shared<LambdaTransform>(func);
        trainSetLoader = std::shared_ptr<DataLoader>(trainDataset.createLoader(trainBatchSize, true, false, trainNumWorkers));
        testSetLoader = std::shared_ptr<DataLoader>(testDataset.createLoader(testBatchSize, true, false, testNumWorkers));
    }
}


