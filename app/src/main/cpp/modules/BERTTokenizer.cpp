//
// Created by 陈宇豪 on 2023/6/15.
//
#include "BERTTokenizer.h"
#include <fstream>
#include <utility>
#include <unicode/uchar.h>

namespace MNN {
    namespace Train {
        namespace Model {
            using namespace MNN::Express;

            std::vector<int> alignLabelsWithTokens(std::vector<int>& labels, std::vector<int>& wordIds) {
                std::vector<int> newLabels;
                int currentWord = WINT_MIN;
                for (auto wordId : wordIds) {
                    if (wordId != currentWord) {
                        // start of a new word
                        currentWord = wordId;
                        int label = wordId == -1 ? -100 : labels[wordId];
                        newLabels.emplace_back(label);
                    } else if (wordId == -1) {
                        // special tokens
                        newLabels.emplace_back(-100);
                    } else {
                        // same word
                        int label = labels[wordId];
                        if (label % 2 == 1) {
                            label += 1;
                        }
                        newLabels.emplace_back(label);
                    }
                }
                return newLabels;
            }

            bool isWhiteSpace(char c) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                    return true;
                }
//                UChar32 uc = static_cast<UChar32>(c);
//                if (u_isWhitespace(uc)) {
//                    return true;
//                }
                return false;
            }

            bool isControl(char c) {
                if (c == '\t' || c == '\n' || c == '\r') {
                    return false;
                }

//                UChar32 uc = static_cast<UChar32>(c);
//                if (u_isISOControl(uc)) {
//                    return true;
//                }
                return false;
            }

            bool isChineseChar(char c) {
                UChar32 uc = static_cast<UChar32>(c);
                if ((uc >= 0x4E00 && uc <= 0x9FFF) ||
                    (uc >= 0x3400 && uc <= 0x4DBF) ||
                    (uc >= 0x20000 && uc <= 0x2A6DF) ||
                    (uc >= 0x2A700 && uc <= 0x2B73F) ||
                    (uc >= 0x2B740 && uc <= 0x2B81F) ||
                    (uc >= 0x2B820 && uc <= 0x2CEAF) ||
                    (uc >= 0xF900 && uc <= 0xFAFF) ||
                    (uc >= 0x2F800 && uc <= 0x2FA1F)) {
                    return true;
                }

                return false;
            }

            bool isPunctuation(char c) {
                UChar32 uc = static_cast<UChar32>(c);
                if ((uc >= 33 && uc <= 47) || (uc >= 58 && uc <= 64) || (uc >= 91 && uc <= 96) || (uc >= 123 && uc <= 126)) {
                    return true;
                }

                return false;
            }

            std::string toLower(std::string text) {
                std::string lowerText;
                for (auto& c : text) {
                    if (c >= 'A' && c <= 'Z') {
                        lowerText.push_back(c + 32);
                    } else {
                        lowerText.push_back(c);
                    }
                }
                return lowerText;
            }


            std::vector<std::string> whitespaceTokenize(std::string& text) {
                std::vector<std::string> tokens;
                std::string token;
                for (auto& c : text) {
                    if (isWhiteSpace(c)) {
                        if (!token.empty()) {
                            tokens.push_back(token);
                            token.clear();
                        }
                    } else {
                        token.push_back(c);
                    }
                }
                if (!token.empty()) {
                    tokens.push_back(token);
                }
                return tokens;
            }

            std::map<std::string, int> loadVocab(const std::string &vocabFile) {
                std::map<std::string, int> vocab;
                std::ifstream in(vocabFile);

                if (!in) {
                    MNN_PRINT("Cannot open %s\n", vocabFile.c_str());
                    return {};
                }
                std::string line;
                int index = 0;
                while (std::getline(in, line)) {

                    vocab[line] = index;
                    index++;
                }
                return vocab;
            }

            BERTTokenizer::BERTTokenizer(std::string &vocalFile, bool doLowerCase, int maxLen, bool doBasicTokenize,
                                         std::vector<std::string> neverSplit): doBasicTokenize(doBasicTokenize) {
                vocab = loadVocab(vocalFile);

                for (auto &item : vocab) {
                    idToTokens[item.second] = item.first;
                }

                if (doBasicTokenize) {
                    basicTokenizer = std::make_shared<BasicTokenizer>(doLowerCase, std::move(neverSplit));
                }
                wordPieceTokenizer = std::make_shared<WordPieceTokenizer>(vocab);

                this->maxLen = maxLen == -1 ? 512 : maxLen;
            }

            std::pair<std::vector<std::string>, std::vector<int>> BERTTokenizer::tokenize(std::string &text) {
                std::vector<std::string> splitTokens;
                std::vector<int> tokenIds;
                // The tokenIds is used to record the index of the original token
                int idx = 0;
                if (doBasicTokenize) {
                    for (auto& token : basicTokenizer->tokenize(text)) {
                        for (auto& subToken : wordPieceTokenizer->tokenize(token)) {
                            splitTokens.push_back(subToken);
                            tokenIds.push_back(idx);
                        }
                        idx++;
                    }
                } else {
                    splitTokens = whitespaceTokenize(text);
                    for (int i = 0; i < splitTokens.size(); i++) {
                        tokenIds.push_back(i);
                    }
                }

                return {splitTokens, tokenIds};
            }

            VARP BERTTokenizer::convertTokensToIds(std::vector<std::string> &tokens, bool padding) {
                std::vector<int> ids;
                for (auto& token : tokens) {
                    ids.push_back(vocab[token]);
                }

                if (padding) {
                    for (int i = tokens.size(); i < maxLen; i++) {
                        ids.push_back(0);
                    }
                }

                // Convert to VARP Tensor
                int tokenLen = ids.size();
                auto inputIds = _Input({1, tokenLen}, NCHW, halide_type_of<int>());
                auto inputPtr = inputIds->writeMap<int>();
                ::memcpy(inputPtr, ids.data(), ids.size() * sizeof(int));
                return inputIds;
            }

            BasicTokenizer::BasicTokenizer(bool doLowerCase, std::vector<std::string> neverSplit) {
                this->doLowerCase = doLowerCase;
                this->neverSplit = std::move(neverSplit);
            }

            std::vector<std::string> BasicTokenizer::tokenize(std::string &text) {
                text = cleanText(text);

                // TODO: Since chinese char should be stored in wchar_t type, we ignore now
                // text = tokenizeChineseChars(text);

                auto originTokens = whitespaceTokenize(text);
                std::vector<std::string> splitTokens;
                for (auto& token : originTokens) {
                    if (doLowerCase && !neverSplit.empty() && std::find(neverSplit.begin(), neverSplit.end(), token) == neverSplit.end()) {
                        token = toLower(token);
                        // accents strip is not considered now
                    }
                    auto subTokens = runSplitOnPunc(token);
                    for (auto& subToken : subTokens) {
                        splitTokens.push_back(subToken);
                    }
                }

                std::vector<std::string> outputTokens;

                std::string splitTokensStr;
                for (auto& token : splitTokens) {
                    splitTokensStr += token;
                    splitTokensStr += " ";
                }

                outputTokens = whitespaceTokenize(splitTokensStr);
                return outputTokens;
            }

            std::string BasicTokenizer::cleanText(std::string &text) {
                std::string output = "";
                for (auto& c : text) {
                    // get the code point of c
                    UChar32 cp = static_cast<UChar32>(c);
                    if (cp == 0 || cp == 0xfffd || isControl(c)) {
                        continue;
                    }

                    if (isWhiteSpace(c)) {
                        output += " ";
                    } else {
                        output += c;
                    }
                }

                return output;
            }

            std::string BasicTokenizer::tokenizeChineseChars(std::string &text) {
                std::string output = "";
                for (auto& c : text) {
                    if (isChineseChar(c)) {
                        output += " ";
                        output += c;
                        output += " ";
                    } else {
                        output += c;
                    }
                }
                return output;
            }

            std::vector<std::string> BasicTokenizer::runSplitOnPunc(std::string &text) {
                std::vector<std::string> tokens;
                int startNewWord = 0;
                for (int i = 0; i < text.size(); i++) {
                    auto c = text[i];
                    if (isPunctuation(c)) {
                        tokens.push_back(text.substr(startNewWord, i - startNewWord));
                        tokens.push_back(std::string(1, c));
                        startNewWord = i + 1;
                    }
                }
                if (startNewWord < text.size()) {
                    tokens.push_back(text.substr(startNewWord, text.size() - startNewWord));
                }
                return tokens;
            }

            WordPieceTokenizer::WordPieceTokenizer(std::map<std::string, int> &vocab, std::string unkToken,
                                                   int maxInputCharsPerWord): vocab(vocab) {
                this->unkToken = unkToken;
                this->maxInputCharsPerWord = maxInputCharsPerWord;
            }

            std::vector<std::string> WordPieceTokenizer::tokenize(std::string &text) {
                std::vector<std::string> outputToken;
                for (auto& token : whitespaceTokenize(text)) {
                    if (token.size() > maxInputCharsPerWord) {
                        outputToken.push_back(unkToken);
                        continue;
                    }

                    bool isBad = false;
                    int start = 0;
                    std::vector<std::string> subTokens;
                    while (start < token.size()) {
                        int end = token.size();
                        std::string curSubStr;
                        while (start < end) {
                            auto substr = token.substr(start, end - start);
                            if (start > 0) {
                                substr = "##" + substr;
                            }
                            if (vocab.find(substr) != vocab.end()) {
                                curSubStr = substr;
                                break;
                            }
                            end -= 1;
                        }
                        if (curSubStr.empty()) {
                            isBad = true;
                            break;
                        }
                        subTokens.push_back(curSubStr);
                        start = end;
                    }
                    if (isBad) {
                        outputToken.push_back(unkToken);
                    } else {
                        for (auto& subToken : subTokens) {
                            outputToken.push_back(subToken);
                        }
                    }
                }
                return outputToken;
            }
        }
    }
}