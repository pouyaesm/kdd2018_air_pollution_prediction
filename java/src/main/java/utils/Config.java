package utils;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.common.base.Charsets;
import com.google.common.io.Files;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Static reference to configuration file
 */
public class Config {
    final static String address = "config.json";
    private static HashMap<String, String> config;
    public final static String BEIJING_OBSERVED = "beijingObserved";
    public final static String LONDON_OBSERVED = "londonObserved";
    public final static String SPARK_TEMP = "sparkTemp";
    public final static String SPARK_WAREHOUSE = "sparkWarehouse";
    public final static String HADOOP = "hadoop";

    private static void load(){
        try {
            config = new ObjectMapper()
                    .readValue(Util.read(address),
                            new TypeReference<Map<String, String>>() {});
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String get(String key){
        if(config == null) load();
        return config.getOrDefault(key, "");
    }
}
