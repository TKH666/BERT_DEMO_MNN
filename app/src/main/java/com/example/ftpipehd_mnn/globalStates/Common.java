package com.example.ftpipehd_mnn.globalStates;

import android.app.Activity;
import android.content.Context;

import com.example.ftpipehd_mnn.utils.CustomView;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Semaphore;

public class Common {
    private static String modelName;
    private static Map<String, Double> modelArgs; // Temporarily, the value of the arguments should be double for simplicity

    private static String datasetName;
    private static String datasetPath;
    private static String datasetBasePath;

    private static String basePath;

    private static int batchSize;
    private static List<Integer> inputSize;

    private static int deviceIdx = -1;

    private static Semaphore semaphore;

    public static Map<String, String> workers = new HashMap<String, String>() {{
       put("0", "http://192.168.31.171:50000") ;
    }};

    private static CustomView.LogAdapter logAdapter;
    private static Context context;

    public static List<String> urls = new ArrayList<String>(Arrays.asList("http://192.168.8.102:50000", "http://192.168.8.106:50000"));
    // public static List<String> urls = new ArrayList<String>(Arrays.asList("http://10.0.2.2:8081", "http://10.0.2.2:8082"));
    public static void setModelName(String _modelName) {
        modelName = _modelName;
    }

    public static void setModelArgs(Map<String, Double> args) {
        modelArgs = args;
    }

    public static Map<String, String> getWorkers() {
        // 返回的是引用
        return workers;
    }

    public static List<String> getUrls() {
        return urls;
    }

    public static int getDeviceIdx() {
        return deviceIdx;
    }

    public static String getModelName() {
        return modelName;
    }

    public static Map<String, Double> getModelArgs() {
        return modelArgs;
    }

    public static void setDeviceIdx(int _deviceIdx) {
        deviceIdx = _deviceIdx;
    }

    public static void setWorkers(Map<String, String> _workers) {
        workers = _workers;
    }

    public static void setDatasetName(String _dataset) {
        datasetName = _dataset;
    }

    public static String getDatasetName() {
        return datasetName;
    }

    public static void setDatasetPath(String _path) {
        datasetPath = _path;
    }

    public static String getDatasetPath() {
        return datasetPath;
    }

    public static int getBatchSize() {
        return batchSize;
    }

    public static void setBatchSize(int _batchSize) {
        batchSize = _batchSize;
    }

    public static List<Integer> getInputSize() {
        return inputSize;
    }

    public static void setInputSize(List<Integer> _inputSize) {
        inputSize = _inputSize;
    }

    public static String getDatasetBasePath() {
        return datasetBasePath;
    }

    public static void setDatasetBasePath(String datasetBasePath) {
        Common.datasetBasePath = datasetBasePath;
    }

    public static String getUrlFromWorker(int idx) {
        if (idx == workers.size()) {
            idx = 0;
        }
        return workers.get(String.valueOf(idx));
    }

    public static int getWorkerNum() {
        return workers.size();
    }

    public static void initSemaphore() {
        semaphore = new Semaphore(getWorkerNum());
        // semaphore = new Semaphore(1);
    }

    public static Semaphore getSemaphore() {
        return semaphore;
    }

    public static void setLogAdapter(Context _context, CustomView.LogAdapter _logAdapter) {
        logAdapter = _logAdapter;
        context = _context;
    }

    public static void printLog(String message) {
        Activity activity = (Activity) context;
       activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                logAdapter.addLog(new CustomView.LogItem(message));
            }
        });
    }

    public static void setBasePath(String _basePath) {
        basePath = _basePath;
    }

    public static String getBasePath() {
        return basePath;
    }
}
