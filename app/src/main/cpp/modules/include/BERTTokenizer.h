//
// Created by 陈宇豪 on 2023/6/15.
//

#ifndef MNN_BERTTOKENIZER_H
#define MNN_BERTTOKENIZER_H

#include <MNN/expr/Module.hpp>
#include "NN.hpp"

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace Express;

            std::map<std::string, int> loadVocab(const std::string &vocabFile);
            std::vector<int> alignLabelsWithTokens(std::vector<int>& labels, std::vector<int>& wordIds);

            class MNN_PUBLIC BasicTokenizer {
            private:
                bool doLowerCase;
                std::vector<std::string> neverSplit;
            public:
                BasicTokenizer(bool doLowerCase = true, std::vector<std::string> neverSplit = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"});
                std::vector<std::string> tokenize(std::string& text);
                std::string cleanText(std::string& text);
                std::string tokenizeChineseChars(std::string& text);
                std::vector<std::string> runSplitOnPunc(std::string& text);
            };

            class MNN_PUBLIC WordPieceTokenizer {
            private:
                std::map<std::string, int>& vocab;
                std::string unkToken;
                int maxInputCharsPerWord;
            public:
                WordPieceTokenizer(std::map<std::string, int>& vocab, std::string unkToken = "[UNK]", int maxInputCharsPerWord = 100);
                std::vector<std::string> tokenize(std::string& text);
            };

            class MNN_PUBLIC BERTTokenizer {
            private:
                std::map<std::string, int> vocab;
                std::map<int, std::string> idToTokens;
                std::shared_ptr<BasicTokenizer> basicTokenizer;
                std::shared_ptr<WordPieceTokenizer> wordPieceTokenizer;
                bool doLowerCase;
                int maxLen;
                bool doBasicTokenize;
            public:
                explicit BERTTokenizer(std::string& vocalFile, bool doLowerCase = true, int maxLen = -1, bool doBasicTokenize = true, std::vector<std::string> neverSplit = {"[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"});
                std::pair<std::vector<std::string>, std::vector<int>> tokenize(std::string& text);
                VARP convertTokensToIds(std::vector<std::string>& tokens, bool padding = true);
            };

        }
    }
}

#endif //MNN_BERTTOKENIZER_H
