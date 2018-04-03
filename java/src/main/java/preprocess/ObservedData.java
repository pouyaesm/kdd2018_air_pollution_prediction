package preprocess;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.StructType;
import scala.Tuple2;
import utils.SparkSetup;

import java.util.ArrayList;

public class ObservedData {

    private Dataset<Row> records;

    public final static String STATION_ID = "station_id";
    public final static String TIME = "utc_time";
    public final static String PM25 = "PM2.5";
    public final static String PM10 = "PM10";
    public final static String O3 = "O3";

    public final static StructType SCHEMA = new StructType()
            .add(STATION_ID, "string")
            .add(TIME, "string")
            .add(PM25, "float")
            .add(PM10, "float")
            .add(O3, "float");

    public ObservedData load(String address){
        SparkSession sparkSession = SparkSetup.getSession();
        this.records = sparkSession.read()
                .format("csv")
                .option("mode", "PERMISSIVE")
                .option("header", "true")
                .option("delimiter", ";")
                .schema(SCHEMA)
                .load(address);
        return this;
    }

    /**
     * Sort the data-set by timestamp ascending
     * @return
     */
    public ObservedData sort(){
        records = records.sort(functions.col("utc_time").asc_nulls_last());
        return this;
    }

    /**
     * Group each station into a list of rows keyed by station id
     * @return
     */
    public ObservedPair group(){
        JavaRDD<Row> rdd = records.toJavaRDD();
        // aggregate rows of a station id into a list of rows per station
        JavaPairRDD<String, ArrayList<ObservedRow>> stationRDD = rdd.mapToPair(
                // row -> (stationId, record)
                row -> {
                    ObservedRow record = new ObservedRow(row);
                    return new Tuple2<>(row.get(0).toString(), record);
                }
        ).aggregateByKey(new ArrayList<>(),
                (ArrayList<ObservedRow> list, ObservedRow record)
                        -> { list.add(record);return list; },
                (ArrayList<ObservedRow> list1, ArrayList<ObservedRow> list2)
                        -> {list1.addAll(list2); return list1;});
        return new ObservedPair(stationRDD);
    }

    /**
     * Get column index of field name in data-set
     * @param fieldName
     * @return
     */
    public static int index(String fieldName){
        return SCHEMA.fieldIndex(fieldName);
    }

    public Dataset<Row> get(){
        return records;
    }
}
