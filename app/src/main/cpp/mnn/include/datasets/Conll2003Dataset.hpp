//
// Created by 陈宇豪 on 2023/6/26.
//

#ifndef MNN_CONLL2003DATASET_H
#define MNN_CONLL2003DATASET_H

#include <string>
#include "Dataset.hpp"
#include "Example.hpp"
#include <fstream>
#include "BERTTokenizer.hpp"

namespace MNN {
    namespace Train {
        struct Conll2003Data {
            int guid;
            std::vector<std::string> tokens;
            std::vector<std::string> posTags;
            std::vector<std::string> chunkTags;
            std::vector<std::string> nerTags;
            std::vector<int> posTagsId;
            std::vector<int> chunkTagsId;
            std::vector<int> nerTagsId;
        };

        class MNN_PUBLIC Conll2003Dataset : public Dataset {
        public:
            enum Mode { TRAIN, TEST };

            Example get(size_t index) override;

            Conll2003Data getOneData();
            size_t size() override;

            void readDataset(const std::string& root, bool train = true);
            static DatasetPtr create(const std::string path, Mode mode = Mode::TRAIN);
            int maxLen;
        private:
            explicit Conll2003Dataset(const std::string path, Mode mode = Mode::TRAIN);
            std::ifstream dataFile;
            int guid;
            bool isTrain;
            std::shared_ptr<Model::BERTTokenizer> bertTokenizer;
            std::vector<VARP> vocabIds;
            std::vector<VARP> nerIds;
        };
    }
}

#endif //MNN_CONLL2003DATASET_H
