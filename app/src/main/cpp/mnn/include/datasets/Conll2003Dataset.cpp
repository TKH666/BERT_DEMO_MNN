//
// Created by 陈宇豪 on 2023/6/26.
//

#include "Conll2003Dataset.hpp"
#include <string>
#include <sstream>
#include <MNN/expr/Module.hpp>
#include <iterator>


namespace MNN {
    namespace Train {
        // referenced from huggingface/datasets
        // https://huggingface.co/datasets/conll2003/blob/main/conll2003.py
        const int32_t kTrainSize = 14041;
        const int32_t kTestSize = 3453;
        const int32_t kValidSize = 3250;

        const char* trainFileName = "conll2003/train.txt";
        const char* testFileName = "conll2003/test.txt";
        const char* validFileName = "conll2003/valid.txt";

        std::vector<std::string> posNames = {"\"", "''", "#", "$", "(", ")", ",", ".", ":", "``", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NN", "NNP", "NNPS", "NNS", "NN|SYM", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS","RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB"};
        std::vector<std::string> chunkNames = {"O", "B-ADJP", "I-ADJP", "B-ADVP", "I-ADVP", "B-CONJP", "I-CONJP", "B-INTJ", "I-INTJ", "B-LST", "I-LST", "B-NP", "I-NP", "B-PP", "I-PP", "B-PRT", "I-PRT", "B-SBAR", "I-SBAR", "B-UCP", "I-UCP", "B-VP", "I-VP"};
        std::vector<std::string> nerNames = {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"};

        std::string conll2023JoinPaths(std::string head, const std::string& tail) {
            if (head.back() != '/') {
                head.push_back('/');
            }
            head += tail;
            return head;
        }

        void Conll2003Dataset::readDataset(const std::string& root, bool train) {
            using namespace Model;
            const auto path = conll2023JoinPaths(root, train ? trainFileName : testFileName);
            std::string vocabPath = conll2023JoinPaths(root, "vocab_files/bert-base-cased-vocab.txt");


            this->maxLen = 180;
            // auto bertTokenizer = Model::BERTTokenizer(vocabPath, false, 256);
            bertTokenizer.reset(new Model::BERTTokenizer(vocabPath, false, maxLen));

            this->dataFile.open(path, std::ios::binary);
            // std::ifstream textData(path, std::ios::binary);
            if (!this->dataFile.is_open()) {
                MNN_PRINT("Error opening dataset file at %s", path.c_str());
                MNN_ASSERT(false);
            }

            // Convert all data to ids
//            for (int i = 0; i < kTrainSize; i++) {
//                auto curData = getOneData();
//                std::string input = "";
//
//                for (auto &word: curData.tokens) {
//                    input += word + " ";
//                }
//                // trailing space
//                input.pop_back();
//
//                auto [tokens, ids] = bertTokenizer.tokenize(input);
//
//                // add [CLS] and [SEP] to the tokens
//                tokens.insert(tokens.begin(), "[CLS]");
//                tokens.emplace_back("[SEP]");
//
//                // add -1 to the ids
//                ids.insert(ids.begin(), -1);
//                ids.emplace_back(-1);
//
//                auto vocabId = bertTokenizer.convertTokensToIds(tokens);
//                vocabId = _Squeeze(vocabId, {0});
//                vocabIds.emplace_back(vocabId);
//
//                // align tokens and convert to VARP
//                auto alignedId = alignLabelsWithTokens(curData.nerTagsId, ids);
//                auto alignedIdVARP = _Input({}, NCHW, halide_type_of<int>());
//                auto inputPtr = alignedIdVARP->writeMap<int>();
//                ::memcpy(inputPtr, alignedId.data(), alignedId.size() * sizeof(int));
//
//                nerIds.emplace_back(alignedIdVARP);
//            }
        }

        DatasetPtr Conll2003Dataset::create(const std::string path, Mode mode) {
            DatasetPtr res;
            res.mDataset.reset(new Conll2003Dataset(path, mode));
            return res;
        }

        Conll2003Dataset::Conll2003Dataset(const std::string path, MNN::Train::Conll2003Dataset::Mode mode) {
            this->isTrain = mode == Mode::TRAIN;
            this->guid = 0;
            readDataset(path, mode == Mode::TRAIN);
        }

        Example Conll2003Dataset::get(size_t index) {
            using namespace Model;
            auto curData = getOneData();
            std::string input = "";

            for (auto &word: curData.tokens) {
                input += word + " ";
            }
            // trailing space
            input.pop_back();

            auto [tokens, ids] = bertTokenizer->tokenize(input);

            // add [CLS] and [SEP] to the tokens
            tokens.insert(tokens.begin(), "[CLS]");
            tokens.emplace_back("[SEP]");

            // add -1 to the ids
            ids.insert(ids.begin(), -1);
            ids.emplace_back(-1);

            auto vocabId = bertTokenizer->convertTokensToIds(tokens);
            vocabId = _Squeeze(vocabId, {0});

            // align tokens and convert to VARP
            auto alignedId = alignLabelsWithTokens(curData.nerTagsId, ids);
            // pad the alignedId
            for (int i = alignedId.size(); i < maxLen; i++) {
                alignedId.emplace_back(-100);
            }

            auto alignedIdVARP = _Input({maxLen}, NCHW, halide_type_of<int>());
            auto inputPtr = alignedIdVARP->writeMap<int>();
            ::memcpy(inputPtr, alignedId.data(), alignedId.size() * sizeof(int));

            auto returnIndex = _Const(index);
            return {{vocabId, returnIndex}, {alignedIdVARP}};
        }
//
        size_t Conll2003Dataset::size() {
            return isTrain ? kTrainSize : kTestSize;
        }

        Conll2003Data Conll2003Dataset::getOneData() {
            if (guid >= kTrainSize) {
                guid = 0;
                this->dataFile.clear();
                this->dataFile.seekg(0, std::ios::beg);
            }

            std::string line;
            Conll2003Data cur = {};
            cur.guid = guid;

            while (std::getline(this->dataFile, line)) {
                if (line.find("-DOCSTART-") == 0 || line == "" || line == "\n") {
                    if (!cur.tokens.empty()) {
                        guid++;
                        return cur;
                    }
                } else {
                    // split line by space
                    std::istringstream iss(line);
                    std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                                     std::istream_iterator<std::string>());
                    cur.tokens.push_back(results[0]);
                    // cur.posTags.push_back(results[1]);
                    // cur.posTagsId.push_back(std::find(posNames.begin(), posNames.end(), results[1]) - posNames.begin());

                    // cur.chunkTags.push_back(results[2]);
                    // cur.chunkTagsId.push_back(std::find(chunkNames.begin(), chunkNames.end(), results[2]) - chunkNames.begin());
                    // strip the trailing whitespace
                    results[3].erase(std::remove(results[3].begin(), results[3].end(), '\n'), results[3].end());
                    cur.nerTags.push_back(results[3]);
                    cur.nerTagsId.push_back(std::find(nerNames.begin(), nerNames.end(), results[3]) - nerNames.begin());
                }
            }

            // last example
            if (!cur.tokens.empty()) {
                guid++;
                return cur;
            }

            return cur;
        }
    }
}