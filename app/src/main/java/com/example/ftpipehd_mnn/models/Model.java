package com.example.ftpipehd_mnn.models;

import com.example.ftpipehd_mnn.globalStates.Common;
import java.util.Map;

public class Model {
    private static final String tag = "Model";
    static {
        System.loadLibrary("ftpipehd_mnn");
    }

    public static native void initModel(String name, Map<String, Double> args);

    public static void createModel() {
        initModel(Common.getModelName(), Common.getModelArgs());
    }

    public static native void initTrain(int batchSize);
    public static native void singleTrainOneEpoch(int epoch);

}
