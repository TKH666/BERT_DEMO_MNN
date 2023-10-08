//
// Created by 陈宇豪 on 2023/2/21.
//

#ifndef FTPIPEHD_MNN_JNIUTILS_H
#define FTPIPEHD_MNN_JNIUTILS_H
#include <jni.h>
#include <string>
#include <map>
#include "MNN/expr/Module.hpp"

using namespace MNN;

namespace FTPipeHD {
    void JavaHashMapToStlStringDoubleMap(JNIEnv *env, jobject hashMap, std::map<std::string, double>& mapOut);
    jobjectArray convertVARPintoObjectArr(JNIEnv *env, Express::VARP output);
    jobjectArray convertVARPPairIntoObjectArr(JNIEnv *env, std::pair<std::vector<Express::VARP>, Express::VARP>& output);
    jobjectArray convertCentralForwardIntoObjectArr(JNIEnv *env, std::pair<std::vector<Express::VARP>, std::vector<Express::VARP> >& output);
    jobjectArray convertFloatVARPPairIntoObjectArr(JNIEnv *env, std::pair<float, Express::VARP>& output);

    std::vector<Express::VARP> convertObjectArrIntoVARPs(JNIEnv *env, jobject input_datas, jobject dimArrs,
                                                   jobject orders);
    std::vector<Express::VARP> convertIntObjectArrIntoVARPs(JNIEnv *env, jobject input_datas, jobject dimArrs,
                                                      jobject orders);
    Express::VARP convertObjectIntoVARP(JNIEnv *env, jintArray input_data, jobject dimArr, jint order);
}

#endif //FTPIPEHD_MNN_JNIUTILS_H
