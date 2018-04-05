package preprocess;

import org.apache.spark.api.java.JavaPairRDD;

import java.util.ArrayList;

public class ObsPair {
    /**
     * Each key is a station id with time-sorted array of measurement reports
     */
    JavaPairRDD<String, ArrayList<BJRow>> data;

    public ObsPair(JavaPairRDD<String, ArrayList<BJRow>> data){
        this.data = data;
    }

    /**
     * Fill missing values with the average of nearest values
     * Assuming the lists are sorted ascending by time
     * @return
     */
    public ObsPair fill(){
        data.mapValues(list -> {
            list =  BJRow.fill(list, new int[]{
                    KDDRow.index(KDDRow.PM25), KDDRow.index(KDDRow.PM10), KDDRow.index(KDDRow.O3)});
            return list;
        });
        return this;
    }

    public JavaPairRDD<String, ArrayList<BJRow>> get(){
        return data;
    }
}
