package com.example.ftpipehd_mnn.globalStates;

import android.content.Context;
import android.util.Log;

import androidx.annotation.NonNull;

import com.example.ftpipehd_mnn.R;

import org.yaml.snakeyaml.Yaml;

import java.io.InputStream;
import java.util.List;
import java.util.Map;

public class Config {
    public static Yaml yaml;
    public static Map<String, Object> cfg;
    public static void loadConfig(@NonNull Context context) {
        InputStream inputStream = context.getResources().openRawResource(R.raw.bert_config);
        yaml = new Yaml();
        cfg = yaml.load(inputStream);

        // model
        Common.setModelName((String) cfg.get("model_name"));
        Common.setModelArgs((Map<String, Double>) cfg.get("model_args"));
        Training.setAggregateInterval((int) cfg.get("weight_aggregation_interval"));

        // optimizer
        Map<String, Object> schedule = (Map<String, Object>) cfg.get("schedule");
        Training.setOptName((String) schedule.get("opt_name"));
        Training.setOptArgs((Map<String, Double>) schedule.get("opt_args"));

        // datasets
        Map<String, Object> datasets = (Map<String, Object>) cfg.get("data");
        Common.setDatasetName((String) datasets.get("name"));
        Common.setDatasetPath((String) datasets.get("path"));
        Common.setBatchSize((int) datasets.get("batch_size"));
        Common.setInputSize((List<Integer>) datasets.get("input_size"));

        Training.setEpochs((int) schedule.get("total_epochs"));

        // pretrain weights
        if (cfg.get("weights_path") != null) {
            Training.setWeightsPath((String) cfg.get("weights_path"));
        }
    }


}
