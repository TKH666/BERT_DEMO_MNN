package com.example.ftpipehd_mnn;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import com.example.ftpipehd_mnn.databinding.ActivityMainBinding;

import com.example.ftpipehd_mnn.datasets.Dataset;
import com.example.ftpipehd_mnn.globalStates.Common;
import com.example.ftpipehd_mnn.globalStates.Config;

import com.example.ftpipehd_mnn.utils.CustomView;
import com.example.ftpipehd_mnn.utils.General;

import com.example.ftpipehd_mnn.utils.TrainSingle;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'ftpipehd_mnn' library on application startup.
    static {
        System.loadLibrary("ftpipehd_mnn");
    }

    private ActivityMainBinding binding;
    private ImageView imageView;
    private RecyclerView logView;
    private CustomView.LogAdapter logAdapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        Common.setBasePath(this.getFilesDir().toString());

        initView();
        initButton();

        int permission = ActivityCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE);
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE, Manifest.permission.WRITE_EXTERNAL_STORAGE}, 222);
            finish();
        }
    }

    /**
     * Init the UI
     */
    private void initView() {
        logView = findViewById(R.id.logRecycler);

        LinearLayoutManager layoutManager = new LinearLayoutManager(this);
        logView.setLayoutManager(layoutManager);

        layoutManager.setOrientation(LinearLayoutManager.HORIZONTAL);
        layoutManager.setOrientation(LinearLayoutManager.VERTICAL);

        List<CustomView.LogItem> logs = new ArrayList<>();
        logAdapter = new CustomView.LogAdapter(logs);
        Common.setLogAdapter(MainActivity.this, logAdapter);
        logView.setAdapter(logAdapter);
    }

    private void initButton() {
        Button singleBtn = findViewById(R.id.single_btn);

        singleBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                singleBtn.setVisibility(View.GONE);
                Common.printLog("Init single training ...");

                copyDatasetsFromAssets();
                initConfig();

                singleTrain();
            }
        });
    }

    private void singleTrain() {
        Thread thread = new Thread(new Runnable() {
            @Override
            public void run() {
                TrainSingle.startTrain();
            }
        }, "singleTraining");
        thread.start();
    }


    /**
     * Load the config file and initialize the config
     */
    private void initConfig() {
        Common.printLog("Loading the config file ... ");
        Config.loadConfig(MainActivity.this);
    }


    /**
     * Copy the datasets to the app's directories from the assets
     */
    private void copyDatasetsFromAssets() {
        String dirName = "trainingData";
        String datasetBasePath = this.getFilesDir() + File.separator + "trainingData" + File.separator;
        Common.setDatasetBasePath(datasetBasePath);

        Common.printLog("Copy datasets to current directory...");
        try {
            General.copyAssetDirToFiles(this, dirName);
            Common.printLog("Copy datasets success...");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Write Permission!", Toast.LENGTH_SHORT).show();
                this.finish();
            }
        }
    }

}