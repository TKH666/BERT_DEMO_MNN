package com.example.ftpipehd_mnn.utils;

import android.util.Log;

import com.example.ftpipehd_mnn.datasets.Dataset;
import com.example.ftpipehd_mnn.globalStates.Backend;
import com.example.ftpipehd_mnn.globalStates.Common;
import com.example.ftpipehd_mnn.globalStates.Training;
import com.example.ftpipehd_mnn.models.Model;

import java.util.Map;

public class TrainSingle {
    private static final String tag = "TrainSingle";

    public static void startTrain() {
        Common.printLog("Single train start");

        Common.printLog(String.format("Creating model %s...", Common.getModelName()));
        Model.createModel();

        Common.printLog("Setting up dataset: " + Common.getDatasetName() + " with batch size " + Common.getBatchSize());
        Dataset.initDataset(Common.getDatasetName(), Common.getDatasetPath(), Common.getBatchSize());

        Log.i(tag, "Initializing executor and batch size in jni ...");
        Model.initTrain(Common.getBatchSize());

        TrainSingle();
    }

    public static void TrainSingle() {
        Log.i(tag, "Start formal single training");

        long startTime = System.currentTimeMillis();
        int dataLen = Dataset.getDataLen();
        int epochs = 1;

        Map<Integer, Integer> backends = Backend.getBackendsMap();
        for (Map.Entry<Integer, Integer> entry : backends.entrySet()) {
            Common.printLog(String.format("Backend %d, NumThreads %d", entry.getKey(), entry.getValue()));
        }

        for (int epoch = 0; epoch < epochs; epoch++) {
            Model.singleTrainOneEpoch(epoch);
        }
        Common.printLog(String.format("Finish, time:%f s", (System.currentTimeMillis() - startTime)/1000.0));
    }
}
