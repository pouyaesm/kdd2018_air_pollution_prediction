package depricated;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.*;
import org.apache.spark.sql.types.StructType;
import static org.apache.spark.sql.functions.regexp_replace;

import scala.Tuple2;
import utils.Config;
import utils.SparkSetup;
import utils.Util;

import java.util.ArrayList;
import java.util.HashMap;

public class BJFiles {

    private Dataset<Row> records;
    private Dataset<Row> airQuality;
    private Dataset<Row> meteorology;
    private Dataset<Row> stations;
    private Dataset<Row> all;
    private Dataset<Row> all2;
    private HashMap<String, String> addresses;

    // The order of field additions must match the order of fields in the files!
    public final static StructType SCHEMA_AQ = new StructType()
            .add(KDDRow.STATION_ID2, "string").add(KDDRow.TIME, "string")
            .add(KDDRow.PM25, "float").add(KDDRow.PM10, "float")
            .add(KDDRow.NO2, "float").add(KDDRow.CO, "float")
            .add(KDDRow.O3, "float").add(KDDRow.SO2, "float");

    public final static StructType SCHEMA_MEO = new StructType()
            .add(KDDRow.STATION_ID, "string").add(KDDRow.LONG, "float")
            .add(KDDRow.LAT, "float").add(KDDRow.TIME, "string")
            .add(KDDRow.TEMP, "float").add(KDDRow.PRES, "float")
            .add(KDDRow.HUM, "float").add(KDDRow.W_DIR, "float")
            .add(KDDRow.W_SPEED, "float").add(KDDRow.WEATHER, "string");

    public final static StructType SCHEMA_STATIONS = new StructType()
            .add(KDDRow.STATION_ID, "string").add(KDDRow.LONG, "float")
            .add(KDDRow.LAT, "float").add(KDDRow.STATION_TYPE, "string");

    public BJFiles(){
        addresses = new HashMap<>();
    }


    /**
     * Set addresses for raw data to be loaded
     * @param key
     * @param address
     * @return
     */
    public BJFiles set(String key, String address){
        addresses.put(key, address);
        return this;
    }

    public BJFiles load(String address){
        SparkSession sparkSession = SparkSetup.getSession();
        this.records = sparkSession.read()
                .format("csv")
                .option("mode", "PERMISSIVE")
                .option("header", "true")
                .option("delimiter", ";")
                .schema(SCHEMA_AQ)
                .load(address);
        return this;
    }


    public BJFiles load(){
        BJRow converter = new BJRow();
        Dataset<Row> aQ = Util.read(addresses.get(Config.BEIJING_OBS_AQ), SCHEMA_AQ)
                .union(Util.read(addresses.get(Config.BEIJING_OBS_AQ_REST), SCHEMA_AQ));
        Dataset<Row> meo = Util.read(addresses.get(Config.BEIJING_OBS_MEO), SCHEMA_MEO);
        Dataset<Row> stations = Util.read(addresses.get(Config.BEIJING_STATIONS), SCHEMA_STATIONS);
        String ID = KDDRow.STATION_ID;
        String TIME = KDDRow.TIME;
        // stations that can report air quality
//        aQ = aQ.withColumn(KDDRow.IS_AQ, functions.lit(1));
        // rename stationId column to station_id similar to meteorology
        aQ = aQ.withColumnRenamed(KDDRow.STATION_ID2, ID);
        // remove _aq and _meo from station_id to allow joining
        aQ = aQ.withColumn(ID, regexp_replace(aQ.col(ID), "_aq", ""));
        meo = meo.withColumn(ID, regexp_replace(meo.col(ID), "_meo", ""));
        // change invalid values to NULL
        meo = meo.withColumn(ID,
                functions.when(meo.col(KDDRow.TEMP).gt(functions.lit(5)), null));
        this.airQuality = aQ;
        this.meteorology = meo;
        this.stations = stations;
//        this.airQuality = converter.map()
//                .union(converter.map(Util.read(addresses.get(Config.BEIJING_OBS_AQ_REST))));
//        this.meteorology = converter.map(Util.read(addresses.get(Config.BEIJING_OBS_MEO)));
//        this.stations = converter.map(Util.read(addresses.get(Config.BEIJING_STATIONS)));
//        all = aQ.join(meo, aQ.col(ID).equalTo(meo.col(ID))
//                .and(aQ.col(TIME).equalTo(meo.col(TIME))), "outer");
        all = aQ.join(meo, Util.toSeq(ID, TIME), "outer")
                .join(stations, Util.toSeq(ID), "outer")
                .sort(functions.col(TIME).asc_nulls_last());
        return this;
    }

    public BJFiles join(){

        return this;
    }

    /**
     * Sort the data-set by timestamp ascending
     * @return
     */
    public BJFiles sort(){
        records = records.sort(functions.col("utc_time").asc_nulls_last());
        return this;
    }

    /**
     * Group each station into a list of rows keyed by station id
     * @return
     */
    public ObsPair group(){
        JavaRDD<Row> rdd = records.toJavaRDD();
        // aggregate rows of a station id into a list of rows per station
        JavaPairRDD<String, ArrayList<BJRow>> stationRDD = rdd.mapToPair(
                // row -> (stationId, record)
                row -> {
                    BJRow record = new BJRow(row);
                    return new Tuple2<>(row.get(0).toString(), record);
                }
        ).aggregateByKey(new ArrayList<>(),
                (ArrayList<BJRow> list, BJRow record)
                        -> { list.add(record);return list; },
                (ArrayList<BJRow> list1, ArrayList<BJRow> list2)
                        -> {list1.addAll(list2); return list1;});
        return new ObsPair(stationRDD);
    }

    public static String toString(Dataset<Row> dataSet){
        String string = String.join(";", dataSet.columns()).concat("\n");
        Row[] list = (Row[]) dataSet.collect();
        for(Row row : list){
            string = string.concat(row.mkString(";")).concat("\n");
        }
        return string;
    }
    /**
     * Get column index of field name in data-set
     * @param fieldName
     * @return
     */
    public static int index(String fieldName){
        return SCHEMA_AQ.fieldIndex(fieldName);
    }

    public Dataset<Row> get(){
        return records;
    }

    public Dataset<Row> getAll() {
        return all;
    }

    public Dataset<Row> getAll2() {
        return all2;
    }

    public Dataset<Row> getAirQuality() {
        return airQuality;
    }

    public Dataset<Row> getMeteorology() {
        return meteorology;
    }

    public Dataset<Row> getStations() {
        return stations;
    }
}
