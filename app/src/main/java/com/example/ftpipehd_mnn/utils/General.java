package com.example.ftpipehd_mnn.utils;

import android.Manifest;
import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;
import android.util.Pair;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import java.util.ArrayList;

import java.util.List;
import java.util.Map;

public class General {
    static public void copyAssetDirToFiles(Context context, String dirname) throws IOException {
        File dir = new File(context.getFilesDir() + File.separator + dirname);
        dir.mkdir();

        AssetManager assetManager = context.getAssets();
        String[] children = assetManager.list(dirname);
        for (String child : children) {
            child = dirname + File.separator + child;
            String[] grandChildren = assetManager.list(child);
            if (0 == grandChildren.length) {
                copyAssetFileToFiles(context, child);
            } else {
                copyAssetDirToFiles(context, child);
            }
        }
    }

    static public void copyAssetDirToTargetDir(Context context, String src, String dst) throws IOException {
        File dir = new File(context.getFilesDir() + File.separator + dst);
        dir.mkdir();

        AssetManager assetManager = context.getAssets();
        String[] children = assetManager.list(src);
        for (String child : children) {
            String dstChild = dst + child;
            child = src + File.separator + child;
            String[] grandChildren = assetManager.list(child);
            if (0 == grandChildren.length) {
                copyAssetFileToTargetFile(context, child, dstChild);
            } else {
                copyAssetDirToTargetDir(context, child, dstChild);
            }
        }
    }

    static public void copyAssetFileToTargetFile(Context context, String srcFile, String dstFile) throws IOException {
        InputStream is = context.getAssets().open(srcFile);
        byte[] buffer = new byte[is.available()];
        is.read(buffer);
        is.close();

        File of = new File(dstFile);
        of.createNewFile();
        FileOutputStream os = new FileOutputStream(of);
        os.write(buffer);
        os.close();
    }

    static public void copyAssetFileToFiles(Context context, String filename) throws IOException {
        InputStream is = context.getAssets().open(filename);
        byte[] buffer = new byte[is.available()];
        is.read(buffer);
        is.close();

        File of = new File(context.getFilesDir() + File.separator + filename);
        of.createNewFile();
        FileOutputStream os = new FileOutputStream(of);
        os.write(buffer);
        os.close();
    }

    /**
     * Convert the JSONObject to List<Integer> Type
     */
    public static List<Integer> convertJSONToIntegerList(JSONArray listObject) {
        List<Integer> list = new ArrayList<>();

        for (Object obj : listObject) {
            list.add((Integer) obj);
        }
        return list;
    }

    public static Pair<Integer, Integer> getLayerFromPoint(List<Integer> point, int deviceIdx) {
        int workerNum = point.size() + 1;
        if (deviceIdx == 0) {
            return new Pair(0, point.get(0));
        } else if (deviceIdx == workerNum - 1) {
            return new Pair(point.get(deviceIdx - 1) + 1, -1);
        } else {
            return new Pair(point.get(deviceIdx - 1) + 1, point.get(deviceIdx));
        }
    }

}
