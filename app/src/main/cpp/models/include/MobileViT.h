//
// Created by 陈宇豪 on 2023/5/16.
//

#ifndef MobileViT_hpp
#define MobileViT_hpp

#include "MNN/expr/Module.hpp"
#include "NN.hpp"
#include "Initializer.hpp"

namespace MNN {

    class MNN_PUBLIC MobileViT : public Express::Module {
    public:
        MobileViT(std::pair<int, int> imageSize, std::vector<int> dims, std::vector<int> &channels,
                  int numClasses, int expansion = 4, int kernelSize = 3,
                  std::vector<int> patchSize = {2, 2});

        virtual std::vector<Express::VARP>
        onForward(const std::vector<Express::VARP> &inputs) override;

        std::shared_ptr<Express::Module> conv1;
        std::vector<std::shared_ptr<Express::Module> > mv2;
        std::vector<std::shared_ptr<Express::Module> > mvit;
        std::shared_ptr<Express::Module> conv2;
        std::shared_ptr<Express::Module> pool;
        std::shared_ptr<Express::Module> fc;

        std::pair<int, int> imageSize;
    };

}

#endif //MNN_MOBILEVIT_H
