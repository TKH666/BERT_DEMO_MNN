package com.example.ftpipehd_mnn.datasets;

import android.graphics.Bitmap;
import android.util.Log;

import com.example.ftpipehd_mnn.globalStates.Common;

public class Dataset {
    static {
        System.loadLibrary("ftpipehd_mnn");
    }

    public static native void init(String basePath, String name, String path, int batchSize);
    public static native int getDataLen();
    public static void initDataset(String name, String path, int batchSize) {
        String basePath = Common.getDatasetBasePath();
        init(basePath, name, path, batchSize);
    }
}
