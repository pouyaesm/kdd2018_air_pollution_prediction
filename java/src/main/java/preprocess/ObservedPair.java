package preprocess;

import org.apache.spark.api.java.JavaPairRDD;

import java.util.ArrayList;

public class ObservedPair {
    /**
     * Each key is a station id with time-sorted array of measurement reports
     */
    JavaPairRDD<String, ArrayList<ObservedRow>> data;

    public ObservedPair(JavaPairRDD<String, ArrayList<ObservedRow>> data){
        this.data = data;
    }

    /**
     * Fill missing values with the average of nearest values
     * Assuming the lists are sorted ascending by time
     * @return
     */
    public ObservedPair fill(final String[] columns){
        data.mapValues(list -> {
            for(String column : columns) {
                int colIndex = ObservedData.index(column);

            } // each column
            return list;
        });
        return this;
    }

    public JavaPairRDD<String, ArrayList<ObservedRow>> get(){
        return data;
    }
}
