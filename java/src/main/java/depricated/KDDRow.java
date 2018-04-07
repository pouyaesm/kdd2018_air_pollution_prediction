package depricated;

import org.apache.spark.sql.Row;
import utils.Util;

import java.util.ArrayList;
import java.util.HashMap;

public class KDDRow {

    /**
     * Maps column names to index of values and filled arrays
     */
    private static HashMap<String, Integer> mapNameToIndex;

    // Column names in data-sets
    public final static String STATION_ID = "station_id";
    public final static String STATION_ID2 = "stationId";
    public final static String STATION_TYPE = "station_type";
    public final static String LONG = "longitude";
    public final static String LAT = "latitude";
    public final static String TIME = "utc_time";
    public final static String IS_AQ = "is_aq"; // does station report air quality?

    public final static String PM25 = "PM2.5";
    public final static String PM10 = "PM10";
    public final static String O3 = "O3";
    public final static String NO2 = "NO2";
    public final static String CO = "CO";
    public final static String SO2 = "SO2";

    public final static String TEMP = "temperature";
    public final static String PRES = "pressure";
    public final static String HUM = "humidity";
    public final static String W_DIR = "wind_direction";
    public final static String W_SPEED = "wind_speed";
    public final static String WEATHER = "weather";

    public KDDRow(){

    }

    /**
     * Convert list of strings to list of objects
     * First element is assumed to carry column names
     * @param data
     * @return
     */
    public ArrayList<KDDRow> map(ArrayList<String[]> data){
        String[] columns = data.get(0).clone(); // column names
//        // change columns to lower-case (case-insensitive)
//        for(int c = 0 ; c < columns.length ; c++){
//            columns[c] = columns[c].toLowerCase();
//        }
        ArrayList<KDDRow> list = new ArrayList<>(data.size() - 1);
        for(int r = 1 ; r < data.size() ; r++){
            list.add(map(columns, data.get(r)));
        }
        return list;
    }

    /**
     * Convert the given column names and their corresponding values
     * to BJRow object
     * @param columns column names
     * @param values corresponding values per column
     * @return
     */
    public KDDRow map(String [] columns, String[] values){
        return null;
    }

    /**
     * Convert spark sql Row to KDDRow
     * @param row
     * @return
     */
    public KDDRow map(String[] columns, Row row){
        String[] values = new String[columns.length];
        for(int v = 0 ; v < values.length ; v++){
            values[v] = Util.toString(row.get(v));
        }
        return map(columns, values);
    }

    public static int index(String columnName){
        if(mapNameToIndex == null){
            mapNameToIndex = new HashMap<>(15);
            int index = 0;
            mapNameToIndex.put(LONG, index++);
            mapNameToIndex.put(LAT, index++);
            mapNameToIndex.put(PM25, index++);
            mapNameToIndex.put(PM10, index++);
            mapNameToIndex.put(NO2, index++);
            mapNameToIndex.put(CO, index++);
            mapNameToIndex.put(O3, index++);
            mapNameToIndex.put(SO2, index++);
            mapNameToIndex.put(TEMP, index++);
            mapNameToIndex.put(PRES, index++);
            mapNameToIndex.put(HUM, index++);
            mapNameToIndex.put(W_DIR, index++);
            mapNameToIndex.put(W_SPEED, index);
        }
        return Util.toInt(mapNameToIndex.get(columnName));
    }
}
