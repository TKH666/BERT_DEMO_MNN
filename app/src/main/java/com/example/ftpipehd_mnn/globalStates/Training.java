package com.example.ftpipehd_mnn.globalStates;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Training {
    private static int totalLayers;
    private static List<Integer> partitionPoint;

    private static int aggregateInterval = -1;

    private static String optName;
    private static Map<String, Double> optArgs;
    private static int epochs;

    private static Map<String, Object> commit;
    private static final int logInterval = 1;

    private static Map<Integer, Object[]> labelsPool = new HashMap<>();

    private static String weightsPath = "";

    public static void setOptName(String _optName) {
        optName = _optName;
    }

    public static void setOptArgs(Map<String, Double> _args) {
        optArgs = _args;
    }

    public static String getOptName() {
        return optName;
    }

    public static Map<String, Double> getOptArgs() {
        return optArgs;
    }

    public static void setTotalLayers(int _totalLayers) {
        totalLayers = _totalLayers;
    }

    public static int getTotalLayers() {
        return totalLayers;
    }

    public static void setPartitionPoint(List<Integer> _partitionPoint) {
        partitionPoint = _partitionPoint;
    }

    public static List<Integer> getPartitionPoint() {
        return partitionPoint;
    }

    public static int getAggregateInterval() {
        return aggregateInterval;
    }

    public static void setAggregateInterval(int _aggregateInterval) {
        aggregateInterval = _aggregateInterval;
    }

    public static void initCommit() {
        commit = new HashMap<>();
        commit.put("forwardId", -1);
        commit.put("backwardId", -1);
        Lock lock = new ReentrantLock();
        Condition condition = lock.newCondition();
        commit.put("lock", lock);
        commit.put("lockCondition", condition);
        commit.put("epoch", -1);
        commit.put("dataLen", -1);
        // TODO: Training info missed
    }

    public static void setCommitLen(int len) {
        commit.put("dataLen", len);
    }
    public static void setCommitEpoch(int epoch) {
        commit.put("epoch", epoch);
    }
    public static int getEpochs() {
        return epochs;
    }

    public static void setEpochs(int epochs) {
        Training.epochs = epochs;
    }

    public static Map<String, Object> getCommit() {
        return commit;
    }

    public static void updateLabelsPool(int iterId, Object[] labels) {
        labelsPool.put(iterId, labels);
    }

    public static Object[] getLabels(int iterId) {
        return labelsPool.get(iterId);
    }

    public static int getLogInterval() {
        return logInterval;
    }

    public static void setWeightsPath(String path) {
        weightsPath = path;
    }

    public static String getWeightsPath() {
        return weightsPath;
    }
}
