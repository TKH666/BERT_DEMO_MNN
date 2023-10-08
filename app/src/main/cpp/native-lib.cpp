#include <jni.h>
#include <string>

#include "datasets.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>
#include <android/bitmap.h>

#include "general.h"
#include "jniUtils.h"
#include "log.h"

#include "model.h"
#include "RandomGenerator.hpp"

#include "singleTrain.h"
#include "BERT.h"

using namespace MNN;
using namespace MNN::Train::Model;

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ftpipehd_1mnn_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_ftpipehd_1mnn_datasets_Dataset_init(JNIEnv* env, jclass className, jstring basePath, jstring name, jstring path, jint batchSize) {
    using namespace MNN;
    using namespace MNN::Train;

    const char *basePathTemp = env->GetStringUTFChars(basePath, 0);
    const char *nameTemp = env->GetStringUTFChars(name, 0);
    const char *pathTemp = env->GetStringUTFChars(path, 0);

    std::string datasetBasePath = basePathTemp;
    std::string datasetName = nameTemp;
    std::string datasetPath = pathTemp;

    FTPipeHD::Datasets::datasets = new FTPipeHD::Datasets(datasetBasePath, datasetName, datasetPath, batchSize);
}


extern "C"
JNIEXPORT void JNICALL
Java_com_example_ftpipehd_1mnn_models_Model_initModel(JNIEnv *env, jclass clazz, jstring name, jobject args) {
    using namespace MNN;
    using namespace MNN::Train;

    RandomGenerator::generator(17);
    std::string modelName = env->GetStringUTFChars(name, 0);

    std::map<std::string, double> modelArgs;
    FTPipeHD::JavaHashMapToStlStringDoubleMap(env, args, modelArgs);

    FTPipeHD::ModelZoo::createModel(modelName, modelArgs);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_ftpipehd_1mnn_models_Model_singleTrainOneEpoch(JNIEnv *env, jclass clazz, jint epoch) {
    using namespace FTPipeHD;
    singleTrainOneEpoch(epoch);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_ftpipehd_1mnn_models_Model_initTrain(JNIEnv *env, jclass clazz, jint batch_size) {
    using namespace FTPipeHD;
    initTrain(batch_size);
}

extern "C"
JNIEXPORT jobject JNICALL
Java_com_example_ftpipehd_1mnn_globalStates_Backend_getBackendsMap(JNIEnv *env, jclass clazz) {
    // TODO: implement getBackendsMap()
    using namespace FTPipeHD;
    auto exe = Executor::getGlobalExecutor();
    auto attr = exe->getAvailableBackends();

    // Create a new Java HashMap object
    jclass hashMapClass = env->FindClass("java/util/HashMap");
    jmethodID hashMapConstructor = env->GetMethodID(hashMapClass, "<init>", "()V");
    jobject hashMapObj = env->NewObject(hashMapClass, hashMapConstructor);

    // Get the method ID of HashMap.put() method
    jmethodID putMethod = env->GetMethodID(hashMapClass, "put", "(Ljava/lang/Object;Ljava/lang/Object;)Ljava/lang/Object;");

    // Get the method ID of Integer.valueOf() method
    jclass integerClass = env->FindClass("java/lang/Integer");
    jmethodID valueOfMethod = env->GetStaticMethodID(integerClass, "valueOf", "(I)Ljava/lang/Integer;");

    // Get the method ID of Integer.intValue() method
    jmethodID intValueMethod = env->GetMethodID(integerClass, "intValue", "()I");

    for (const auto& pair : attr) {
        jint key = static_cast<jint>(pair.first);
        jint value = static_cast<jint>(pair.second);

        jobject keyObj = env->CallStaticObjectMethod(integerClass, valueOfMethod, key);
        jobject valueObj = env->CallStaticObjectMethod(integerClass, valueOfMethod, value);

        env->CallObjectMethod(hashMapObj, putMethod, keyObj, valueObj);

        env->DeleteLocalRef(keyObj);
        env->DeleteLocalRef(valueObj);
    }

    env->DeleteLocalRef(hashMapClass);
    env->DeleteLocalRef(integerClass);

    return hashMapObj;

}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_ftpipehd_1mnn_datasets_Dataset_getDataLen(JNIEnv *env, jclass clazz) {
    using namespace FTPipeHD;
    return (int) Datasets::trainSetLoader->iterNumber();
}