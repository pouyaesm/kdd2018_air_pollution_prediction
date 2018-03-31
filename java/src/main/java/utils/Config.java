package utils;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Static reference to configuration file
 */
public class Config {
    final static String address = "config.json";
    private static HashMap<String, String> config;
    final static String CLEAN_DATA = "CLEAN_DATA";

    Config(){
        try {
            config = new ObjectMapper()
                    .readValue(address, new TypeReference<Map<String, String>>() {});
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static String get(String key){
        return config.getOrDefault(key, "");
    }
}
