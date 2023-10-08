//
// Created by 陈宇豪 on 2023/1/10.
//

#include "general.h"
#include "log.h"
#include <android/bitmap.h>
#include "MNN/expr/NeuralNetWorkOp.hpp"

using namespace MNN;
using namespace MNN::Express;

namespace FTPipeHD {
    void MatToBitmap2(JNIEnv *env, cv::Mat& mat, jobject& bitmap, jboolean needPremultiplyAlpha) {
        AndroidBitmapInfo info;
        void *pixels = 0;
        cv::Mat &src = mat;

        try {
            //LOGD("nMatToBitmap");
            CV_Assert(AndroidBitmap_getInfo(env, bitmap, &info) >= 0);
            CV_Assert(info.format == ANDROID_BITMAP_FORMAT_RGBA_8888 ||
                      info.format == ANDROID_BITMAP_FORMAT_RGB_565);
            CV_Assert(src.dims == 2 && info.height == (uint32_t) src.rows &&
                      info.width == (uint32_t) src.cols);
            CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3 || src.type() == CV_8UC4);
            CV_Assert(AndroidBitmap_lockPixels(env, bitmap, &pixels) >= 0);
            CV_Assert(pixels);

            if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
                cv::Mat tmp(info.height, info.width, CV_8UC4, pixels);
                if (src.type() == CV_8UC1) {
                    //LOGD("nMatToBitmap: CV_8UC1 -> RGBA_8888");
                    cv::cvtColor(src, tmp, cv::COLOR_GRAY2RGBA);
                }
                else if (src.type() == CV_8UC3) {
                    //LOGD("nMatToBitmap: CV_8UC3 -> RGBA_8888");
                    cv::cvtColor(src, tmp, cv::COLOR_RGB2RGBA);
                }
                else if (src.type() == CV_8UC4) {
                    //LOGD("nMatToBitmap: CV_8UC4 -> RGBA_8888");
                    if (needPremultiplyAlpha) {
                        cv::cvtColor(src, tmp, cv::COLOR_RGBA2mRGBA);
                    }
                    else {
                        src.copyTo(tmp);
                    }
                }
            }
            else {
                // info.format == ANDROID_BITMAP_FORMAT_RGB_565
                cv::Mat tmp(info.height, info.width, CV_8UC2, pixels);
                if (src.type() == CV_8UC1) {
                    //LOGD("nMatToBitmap: CV_8UC1 -> RGB_565");
                    cv::cvtColor(src, tmp, cv::COLOR_GRAY2BGR565);
                }
                else if (src.type() == CV_8UC3) {
                    //LOGD("nMatToBitmap: CV_8UC3 -> RGB_565");
                    cv::cvtColor(src, tmp, cv::COLOR_RGB2BGR565);
                }
                else if (src.type() == CV_8UC4) {
                    //LOGD("nMatToBitmap: CV_8UC4 -> RGB_565");
                    cv::cvtColor(src, tmp, cv::COLOR_RGBA2BGR565);
                }
            }
            AndroidBitmap_unlockPixels(env, bitmap);
            return;
        }
        catch (const cv::Exception &e) {
            AndroidBitmap_unlockPixels(env, bitmap);
            LOGE("nMatToBitmap catched cv::Exception: %s", e.what());
            jclass je = env->FindClass("java/lang/Exception");
            env->ThrowNew(je, e.what());
            env->DeleteLocalRef(je);
            return;
        }
        catch (...) {
            AndroidBitmap_unlockPixels(env, bitmap);
            LOGE("nMatToBitmap catched unknown exception (...)");
            jclass je = env->FindClass("java/lang/Exception");
            env->ThrowNew(je, "Unknown exception in JNI code {nMatToBitmap}");
            env->DeleteLocalRef(je);
            return;
        }
    }

    void MatToBitmap(JNIEnv *env, cv::Mat& mat, jobject& bitmap) {
        MatToBitmap2(env, mat, bitmap, false);
    }

    std::vector<VARP> cloneParams(std::vector<VARP>& params) {
        int n = params.size();
        std::vector<VARP> copiedParams(n);
        for (int i = 0; i < n; i++) {
            copiedParams[i] = _Clone(params[i], true);
//            auto info = params[i]->getInfo();
//            auto ptr = params[i]->readMap<void>();  // 计算并且存入 cache？
//            if (nullptr == ptr) {
//                MNN_ERROR("Compute error in SGD\n");
//                return {};
//            }
//            auto newVar = _Const(ptr, info->dim, info->order, info->type);
//            copiedParams[i]= newVar;
        }
        return copiedParams;
    }
}
