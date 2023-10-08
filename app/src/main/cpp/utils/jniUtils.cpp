//
// Created by 陈宇豪 on 2023/2/21.
//
#include <jniUtils.h>
#include "log.h"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"
#include "MNN/expr/Expr.hpp"

using namespace MNN;

namespace FTPipeHD {
    void JavaHashMapToStlStringDoubleMap(JNIEnv *env, jobject hashMap, std::map<std::string, double>& mapOut) {
        // Get the Map's entry Set.
        jclass mapClass = env->FindClass("java/util/Map");
        if (mapClass == NULL) {
            return;
        }
        jmethodID entrySet =
                env->GetMethodID(mapClass, "entrySet", "()Ljava/util/Set;");
        if (entrySet == NULL) {
            return;
        }
        jobject set = env->CallObjectMethod(hashMap, entrySet);
        if (set == NULL) {
            return;
        }
        // Obtain an iterator over the Set
        jclass setClass = env->FindClass("java/util/Set");
        if (setClass == NULL) {
            return;
        }
        jmethodID iterator =
                env->GetMethodID(setClass, "iterator", "()Ljava/util/Iterator;");
        if (iterator == NULL) {
            return;
        }
        jobject iter = env->CallObjectMethod(set, iterator);
        if (iter == NULL) {
            return;
        }
        // Get the Iterator method IDs
        jclass iteratorClass = env->FindClass("java/util/Iterator");
        if (iteratorClass == NULL) {
            return;
        }
        jmethodID hasNext = env->GetMethodID(iteratorClass, "hasNext", "()Z");
        if (hasNext == NULL) {
            return;
        }
        jmethodID next =
                env->GetMethodID(iteratorClass, "next", "()Ljava/lang/Object;");
        if (next == NULL) {
            return;
        }
        // Get the Entry class method IDs
        jclass entryClass = env->FindClass("java/util/Map$Entry");
        if (entryClass == NULL) {
            return;
        }
        jmethodID getKey =
                env->GetMethodID(entryClass, "getKey", "()Ljava/lang/Object;");
        if (getKey == NULL) {
            return;
        }
        jmethodID getValue =
                env->GetMethodID(entryClass, "getValue", "()Ljava/lang/Object;");
        if (getValue == NULL) {
            return;
        }

        // Iterate over the entry Set
        while (env->CallBooleanMethod(iter, hasNext)) {
            jobject entry = env->CallObjectMethod(iter, next);
            jstring key = (jstring) env->CallObjectMethod(entry, getKey);
            jobject value = env->CallObjectMethod(entry, getValue);

            // convert the value into int type
            jclass intCls = env->FindClass("java/lang/Double");
            if (!intCls) {
                return ;
            }
            jmethodID doubleMethodID = env->GetMethodID(intCls, "doubleValue", "()D");
            if (doubleMethodID == NULL) {
                return ;
            }
            double doubleVal = env->CallDoubleMethod(value, doubleMethodID);

            const char* keyStr = env->GetStringUTFChars(key, NULL);
            if (!keyStr) {  // Out of memory
                return;
            }

            mapOut.insert(std::make_pair(std::string(keyStr), doubleVal));

            env->DeleteLocalRef(entry);
            env->ReleaseStringUTFChars(key, keyStr);
            env->DeleteLocalRef(key);
        }
    }

    jobjectArray convertVARPintoObjectArr(JNIEnv *env, Express::VARP output) {
        auto outputPtr = output->readMap<float>();
        auto outputInfo = output->getInfo();
        auto outputTotalSize = outputInfo->size;
        auto outputSize = outputInfo->dim;

        // data output
        LOGE("Test output: %f", outputPtr[0]);
        jfloatArray data = env->NewFloatArray(outputTotalSize);
        env->SetFloatArrayRegion(data, 0, outputTotalSize, outputPtr);

        // dim vector
        jclass arrayListClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "()V");
        jobject dimVector = env->NewObject(arrayListClass, arrayListConstructor);

        jmethodID arrayListAddMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
        jclass integerClass = env->FindClass("java/lang/Integer");
        jmethodID integerConstructor = env->GetMethodID(integerClass, "<init>", "(I)V");

        for (int i = 0; i < outputSize.size(); i++) {
            jint javaInt = outputSize[i];
            jobject javaIntObj = env->NewObject(integerClass, integerConstructor, javaInt);
            env->CallBooleanMethod(dimVector, arrayListAddMethod, javaIntObj);
        }

        // order
        jint order = outputInfo->order;
        jobject integerObj = env->NewObject(integerClass, integerConstructor, order);

        jobjectArray ret = env->NewObjectArray(3, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(ret, 0, data);
        env->SetObjectArrayElement(ret, 1, dimVector);
        env->SetObjectArrayElement(ret, 2, integerObj);
        return ret;
    }

    jobjectArray convertVARPPairIntoObjectArr(JNIEnv *env, std::pair<std::vector<Express::VARP>, Express::VARP>& output) {
        auto modelOutput = output.first;
        auto labels = output.second;

        auto outputPtr = modelOutput[0]->readMap<float>();
        auto outputInfo = modelOutput[0]->getInfo();
        auto outputTotalSize = outputInfo->size;
        auto outputSize = outputInfo->dim;

        // data output
        LOGE("Test output: %f", outputPtr[0]);
        jfloatArray data = env->NewFloatArray(outputTotalSize);
        env->SetFloatArrayRegion(data, 0, outputTotalSize, outputPtr);

        // dim vector
        jclass arrayListClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "()V");
        jobject dimVector = env->NewObject(arrayListClass, arrayListConstructor);

        jmethodID arrayListAddMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
        jclass integerClass = env->FindClass("java/lang/Integer");
        jmethodID integerConstructor = env->GetMethodID(integerClass, "<init>", "(I)V");

        for (int i = 0; i < outputSize.size(); i++) {
            jint javaInt = outputSize[i];
            jobject javaIntObj = env->NewObject(integerClass, integerConstructor, javaInt);
            env->CallBooleanMethod(dimVector, arrayListAddMethod, javaIntObj);
        }

        // order
        jint order = outputInfo->order;
        jobject integerObj = env->NewObject(integerClass, integerConstructor, order);

        // labels
        auto labelsPtr = _Cast<int32_t>(labels)->readMap<int>();
        auto labelsInfo = labels->getInfo();
        auto labelsTotalSize = labelsInfo->size;
        auto labelsSize = labelsInfo->dim;

        jintArray labelsData = env->NewIntArray(labelsTotalSize);
        env->SetIntArrayRegion(labelsData, 0, labelsTotalSize, labelsPtr);

        // labels dim vector
        jobject labelsDimVector = env->NewObject(arrayListClass, arrayListConstructor);

        for (int i = 0; i < labelsSize.size(); i++) {
            jint javaInt = labelsSize[i];
            jobject javaIntObj = env->NewObject(integerClass, integerConstructor, javaInt);
            env->CallBooleanMethod(labelsDimVector, arrayListAddMethod, javaIntObj);
        }

        // labels order
        jint labelsOrder = labelsInfo->order;
        jobject labelsIntegerObj = env->NewObject(integerClass, integerConstructor, labelsOrder);

        jobjectArray ret = env->NewObjectArray(6, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(ret, 0, data);
        env->SetObjectArrayElement(ret, 1, dimVector);
        env->SetObjectArrayElement(ret, 2, integerObj);
        env->SetObjectArrayElement(ret, 3, labelsData);
        env->SetObjectArrayElement(ret, 4, labelsDimVector);
        env->SetObjectArrayElement(ret, 5, labelsIntegerObj);
        return ret;
    }

    jobjectArray convertCentralForwardIntoObjectArr(JNIEnv *env, std::pair<std::vector<Express::VARP>, std::vector<Express::VARP> >& output) {
        auto modelOutput = output.first;
        auto labels = output.second;

        // basic convertor
        jclass arrayListClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "()V");

        jmethodID arrayListAddMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
        jclass integerClass = env->FindClass("java/lang/Integer");
        jmethodID integerConstructor = env->GetMethodID(integerClass, "<init>", "(I)V");

        // We return an Object[] with 2 Object[] inside
        jobjectArray ret = env->NewObjectArray(2, env->FindClass("java/lang/Object"), NULL);

        // First put the output into an Object[]
        int modelOutputNum = modelOutput.size();
        jobjectArray modelOutputArr = env->NewObjectArray(modelOutputNum * 3, env->FindClass("java/lang/Object"), NULL);
        for (int i = 0; i < modelOutput.size(); i++) {
            auto outputPtr = modelOutput[i]->readMap<float>();
            auto outputInfo = modelOutput[i]->getInfo();
            auto outputTotalSize = outputInfo->size;
            auto outputSize = outputInfo->dim;

            // data output
            jfloatArray data = env->NewFloatArray(outputTotalSize);
            env->SetFloatArrayRegion(data, 0, outputTotalSize, outputPtr);

            // dim vector
            jobject dimVector = env->NewObject(arrayListClass, arrayListConstructor);

            for (int i = 0; i < outputSize.size(); i++) {
                jint javaInt = outputSize[i];
                jobject javaIntObj = env->NewObject(integerClass, integerConstructor, javaInt);
                env->CallBooleanMethod(dimVector, arrayListAddMethod, javaIntObj);
            }

            // order
            jint order = outputInfo->order;
            jobject integerObj = env->NewObject(integerClass, integerConstructor, order);

            env->SetObjectArrayElement(modelOutputArr, i * 3, data);
            env->SetObjectArrayElement(modelOutputArr, i * 3 + 1, dimVector);
            env->SetObjectArrayElement(modelOutputArr, i * 3 + 2, integerObj);
        }

        // Then put the labels into an Object[]
        int labelsNum = labels.size();
        jobjectArray labelsArr = env->NewObjectArray(labelsNum * 3, env->FindClass("java/lang/Object"), NULL);
        for (int i = 0; i < labels.size(); i++) {
            auto labelsPtr = _Cast<int32_t>(labels[i])->readMap<int>();
            auto labelsInfo = labels[i]->getInfo();
            auto labelsTotalSize = labelsInfo->size;
            auto labelsSize = labelsInfo->dim;

            jintArray labelsData = env->NewIntArray(labelsTotalSize);
            env->SetIntArrayRegion(labelsData, 0, labelsTotalSize, labelsPtr);

            // labels dim vector
            jobject labelsDimVector = env->NewObject(arrayListClass, arrayListConstructor);

            for (int i = 0; i < labelsSize.size(); i++) {
                jint javaInt = labelsSize[i];
                jobject javaIntObj = env->NewObject(integerClass, integerConstructor, javaInt);
                env->CallBooleanMethod(labelsDimVector, arrayListAddMethod, javaIntObj);
            }

            // labels order
            jint labelsOrder = labelsInfo->order;
            jobject labelsIntegerObj = env->NewObject(integerClass, integerConstructor, labelsOrder);

            env->SetObjectArrayElement(labelsArr, i * 3, labelsData);
            env->SetObjectArrayElement(labelsArr, i * 3 + 1, labelsDimVector);
            env->SetObjectArrayElement(labelsArr, i * 3 + 2, labelsIntegerObj);
        }

        env->SetObjectArrayElement(ret, 0, modelOutputArr);
        env->SetObjectArrayElement(ret, 1, labelsArr);

        return ret;
    }

    jobjectArray convertFloatVARPPairIntoObjectArr(JNIEnv *env, std::pair<float, Express::VARP>& output) {
        auto loss = output.first;
        auto modelOutput = output.second;

        auto outputPtr = modelOutput->readMap<float>();
        auto outputInfo = modelOutput->getInfo();
        auto outputTotalSize = outputInfo->size;
        auto outputSize = outputInfo->dim;

        // data output
        LOGE("Test output: %f", outputPtr[0]);
        jfloatArray data = env->NewFloatArray(outputTotalSize);
        env->SetFloatArrayRegion(data, 0, outputTotalSize, outputPtr);

        // dim vector
        jclass arrayListClass = env->FindClass("java/util/ArrayList");
        jmethodID arrayListConstructor = env->GetMethodID(arrayListClass, "<init>", "()V");
        jobject dimVector = env->NewObject(arrayListClass, arrayListConstructor);

        jmethodID arrayListAddMethod = env->GetMethodID(arrayListClass, "add", "(Ljava/lang/Object;)Z");
        jclass integerClass = env->FindClass("java/lang/Integer");
        jmethodID integerConstructor = env->GetMethodID(integerClass, "<init>", "(I)V");

        for (int i = 0; i < outputSize.size(); i++) {
            jint javaInt = outputSize[i];
            jobject javaIntObj = env->NewObject(integerClass, integerConstructor, javaInt);
            env->CallBooleanMethod(dimVector, arrayListAddMethod, javaIntObj);
        }

        // order
        jint order = outputInfo->order;
        jobject integerObj = env->NewObject(integerClass, integerConstructor, order);

        // loss value
        jclass floatClass = env->FindClass("java/lang/Float");
        jmethodID floatConstructor = env->GetMethodID(floatClass, "<init>", "(F)V");
        jobject floatObj = env->NewObject(floatClass, floatConstructor, loss);

        jobjectArray ret = env->NewObjectArray(6, env->FindClass("java/lang/Object"), NULL);
        env->SetObjectArrayElement(ret, 0, data);
        env->SetObjectArrayElement(ret, 1, dimVector);
        env->SetObjectArrayElement(ret, 2, integerObj);
        env->SetObjectArrayElement(ret, 3, floatObj);
        return ret;
    }

    std::vector<Express::VARP>
    convertObjectArrIntoVARPs(JNIEnv *env, jobject input_datas, jobject dimArrs,
                                  jobject orders) {
        // get the size of the input_datas
        jclass arrayListClass = env->GetObjectClass(input_datas);
        jmethodID getMethod = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");
        jmethodID sizeMethod = env->GetMethodID(arrayListClass, "size", "()I");

        std::vector<Express::VARP> inputs;
        jint dataSize = env->CallIntMethod(input_datas, sizeMethod);
        for (int i = 0; i < dataSize; i++) {
            // Construct VARPs from input_datas, dims and orders
            // get the float[] from input data
//            jobject input_data = env->CallObjectMethod(input_datas, getMethod, i);
//            jfloat* jFloatArrayElements = env->GetFloatArrayElements((jfloatArray)input_data, nullptr);
//            jsize jFloatArrayLength = env->GetArrayLength((jfloatArray)input_data);
//            float* input = new float[jFloatArrayLength];
//            for (int i = 0; i < jFloatArrayLength; i++) {
//                input[i] = jFloatArrayElements[i];
//            }
//            env->ReleaseFloatArrayElements((jfloatArray)input_data, jFloatArrayElements, 0);
            jobject input_data = env->CallObjectMethod(input_datas, getMethod, i);
            jdouble* jDoubleArrayElements = env->GetDoubleArrayElements((jdoubleArray)input_data, nullptr);
            jsize jDoubleArrayLength = env->GetArrayLength((jdoubleArray)input_data);
            auto input = new float[jDoubleArrayLength];
            for (int i = 0; i < jDoubleArrayLength; i++) {
                input[i] = (float)jDoubleArrayElements[i];
            }
            // get the std::vector<int> from dimArrs
            jobject dimArr = env->CallObjectMethod(dimArrs, getMethod, i);
            jsize jArrayListSize = env->CallIntMethod(dimArr, sizeMethod);
            jclass jIntegerClass = env->FindClass("java/lang/Integer");
            jmethodID jIntValueMethodID = env->GetMethodID(jIntegerClass, "intValue", "()I");

            std::vector<int> dim;
            int size = 1;
            for (int i = 0; i < jArrayListSize; i++) {
                jobject jIntElementObj = env->CallObjectMethod(dimArr, getMethod, i);
                jint jIntElement = (jint) env->CallIntMethod(jIntElementObj, jIntValueMethodID);
                int intElement = (int) jIntElement;
                dim.push_back(intElement);
                size *= intElement;
            }

            // get order from orders
            jobject order = env->CallObjectMethod(orders, getMethod, i);
            jint jOrder = (jint) env->CallIntMethod(order, jIntValueMethodID);
            int intOrder = (int) jOrder;

            // create the VARP var
            auto curInput = _Input(dim, static_cast<Express::Dimensionformat>(intOrder), halide_type_of<float>());
            auto inputPtr = curInput->writeMap<void>();
            ::memcpy(inputPtr, (void*) input, size * halide_type_of<float>().bytes());

            inputs.push_back(curInput);
        }
        return inputs;
    }

    Express::VARP
    convertObjectIntoVARP(JNIEnv *env, jintArray input_data, jobject dimArr, jint order) {
        // jintArray into int array in cpp
        jint* jIntArrayElements = env->GetIntArrayElements(input_data, nullptr);
        jsize jIntArrayLength = env->GetArrayLength(input_data);
        int* inputData = new int[jIntArrayLength];
        for (int i = 0; i < jIntArrayLength; i++) {
            inputData[i] = jIntArrayElements[i];
        }
        env->ReleaseIntArrayElements(input_data, jIntArrayElements, 0);

        jsize jArrayListSize = env->CallIntMethod(dimArr, env->GetMethodID(env->GetObjectClass(dimArr), "size", "()I"));
        jmethodID jArrayListGetMethodID = env->GetMethodID(env->GetObjectClass(dimArr), "get", "(I)Ljava/lang/Object;");
        jclass jIntegerClass = env->FindClass("java/lang/Integer");
        jmethodID jIntValueMethodID = env->GetMethodID(jIntegerClass, "intValue", "()I");

        std::vector<int> dim;
        int size = 1;
        for (int i = 0; i < jArrayListSize; i++) {
            jobject jIntElementObj = env->CallObjectMethod(dimArr, jArrayListGetMethodID, i);
            jint jIntElement = (jint) env->CallIntMethod(jIntElementObj, jIntValueMethodID);
            int intElement = (int) jIntElement;
            dim.push_back(intElement);
            size *= intElement;
        }

        // create the VARP var
        auto var = _Input(dim, static_cast<Express::Dimensionformat>(order), halide_type_of<uint8_t>());
        auto varPtr = var->writeMap<void>();
        ::memcpy(varPtr, (void*) inputData, size * halide_type_of<uint8_t>().bytes());

        return var;
    }

    std::vector<Express::VARP>
    convertIntObjectArrIntoVARPs(JNIEnv *env, jobject input_datas, jobject dimArrs,
                                 jobject orders) {
        // get the size of the input_datas
        jclass arrayListClass = env->GetObjectClass(input_datas);
        jmethodID getMethod = env->GetMethodID(arrayListClass, "get", "(I)Ljava/lang/Object;");
        jmethodID sizeMethod = env->GetMethodID(arrayListClass, "size", "()I");

        std::vector<Express::VARP> inputs;
        jint dataSize = env->CallIntMethod(input_datas, sizeMethod);
        for (int i = 0; i < dataSize; i++) {
            // Construct VARPs from input_datas, dims and orders
            // get the float[] from input data
            jobject input_data = env->CallObjectMethod(input_datas, getMethod, i);
            jint* jIntArrayElements = env->GetIntArrayElements((jintArray)input_data, nullptr);
            jsize jIntArrayLength = env->GetArrayLength((jintArray)input_data);
            int* input = new int[jIntArrayLength];
            for (int i = 0; i < jIntArrayLength; i++) {
                input[i] = jIntArrayElements[i];
            }
            env->ReleaseIntArrayElements((jintArray)input_data, jIntArrayElements, 0);

            // get the std::vector<int> from dimArrs
            jobject dimArr = env->CallObjectMethod(dimArrs, getMethod, i);
            jsize jArrayListSize = env->CallIntMethod(dimArr, sizeMethod);
            jclass jIntegerClass = env->FindClass("java/lang/Integer");
            jmethodID jIntValueMethodID = env->GetMethodID(jIntegerClass, "intValue", "()I");

            std::vector<int> dim;
            int size = 1;
            for (int i = 0; i < jArrayListSize; i++) {
                jobject jIntElementObj = env->CallObjectMethod(dimArr, getMethod, i);
                jint jIntElement = (jint) env->CallIntMethod(jIntElementObj, jIntValueMethodID);
                int intElement = (int) jIntElement;
                dim.push_back(intElement);
                size *= intElement;
            }

            // get order from orders
            jobject order = env->CallObjectMethod(orders, getMethod, i);
            jint jOrder = (jint) env->CallIntMethod(order, jIntValueMethodID);
            int intOrder = (int) jOrder;

            // create the VARP var
            auto curInput = _Input(dim, static_cast<Express::Dimensionformat>(intOrder), halide_type_of<int>());
            auto inputPtr = curInput->writeMap<void>();
            ::memcpy(inputPtr, (void*) input, size * halide_type_of<int>().bytes());

            inputs.push_back(curInput);
        }
        return inputs;
    }


}