//
// Created by 陈宇豪 on 2023/1/10.
//


#ifndef FTPIPEHD_MNN_GENERAL_H
#define FTPIPEHD_MNN_GENERAL_H

#include <jni.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include "MNN/expr/Expr.hpp"

#endif //FTPIPEHD_MNN_GENERAL_H

using namespace MNN;
using namespace MNN::Express;

namespace FTPipeHD {
    void MatToBitmap(JNIEnv *env, cv::Mat& mat, jobject& bitmap);
    void MatToBitmap2(JNIEnv *env, cv::Mat& mat, jobject& bitmap, jboolean needPremultiplyAlpha);
    std::vector<VARP> cloneParams(std::vector<VARP>& params);

}