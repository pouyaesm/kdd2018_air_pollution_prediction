package preprocess;

import org.apache.spark.api.java.JavaPairRDD;

import java.util.ArrayList;

public class KDDDataPair {
    /**
     * Each key is a station id with time-sorted array of measurement reports
     */
    JavaPairRDD<String, ArrayList<Record>> data;

    public KDDDataPair(JavaPairRDD<String, ArrayList<Record>> data){
        this.data = data;
    }

    /**
     * Fill missing values with the average of nearest values
     * Assuming the lists are sorted by time ascending
     * @return
     */
    public KDDDataPair fill(final String[] columns){
        data.mapValues(list -> {
            for(String column : columns) {
                int colIndex = KDDDataSet.index(column);

            } // each column
            return list;
        });
        return this;
    }

    public JavaPairRDD<String, ArrayList<Record>> get(){
        return data;
    }
}
