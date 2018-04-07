package depricated;

import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import utils.MyDateFormat;
import utils.Util;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;

public class BJRow extends KDDRow implements Serializable{

    // Indices in values and filled
//    public final static int I_PM25 = 0;
//    public final static int I_PM10 = 1;
//    public final static int I_O3 = 2;
//    public final static int I_NO2 = 3;
//    public final static int I_CO = 4;
//    public final static int I_SO2 = 5;
//    public final static int I_TEMP = 3;



    public String stationId; // station id
    public String stationType; // station type
    public String utcTime; // UTC time
    public String weather; // weather type
    public int count; // for aggregation purposes

    public float[] values = new float[13];



    /**
     * Indicator of whether the missing value is filled
     */
    public boolean[] filled = new boolean[13];

    public BJRow(){

    }

    public BJRow(String stationId, String utcTime, float PM25, float PM10, float O3){
        init(stationId, utcTime, PM25, PM10, O3);
    }

    BJRow(Row row){
        init(row.getString(BJFiles.index(KDDRow.STATION_ID)),
                row.getString(BJFiles.index(KDDRow.TIME)),
                Util.toFloat(row.get(BJFiles.index(KDDRow.PM25))),
                Util.toFloat(row.get(BJFiles.index(KDDRow.PM10))),
                Util.toFloat(row.get(BJFiles.index(KDDRow.O3))));
    }

    private void init(String stationId, String utcTime, float PM25, float PM10, float O3){
        this.stationId = stationId;
        this.utcTime  = utcTime;
//        this.timestamp = Util.toTime(this.utcTime);
        this.values[index(KDDRow.PM25)] = PM25;
        this.values[index(KDDRow.PM10)] = PM10;
        this.values[index(KDDRow.O3)] = O3;
        this.count = 1;
    }

    /**
     * Fill the value and mark it as filled
     * @param index
     * @param value
     * @return
     */
    public BJRow fill(int index, float value){
        values[index] = value;
        filled[index] = true;
        return this;
    }

    /**
     * Fill in missing (null) values with the average of nearest neighbors
     * @param list
     * @param indices index of values to be filled
     * @return
     */
    public static ArrayList<BJRow> fill(ArrayList<BJRow> list, int[] indices){
        int size = list.size();
        for(int index: indices) {
            int first = -1; // first null value
            int last = -1; // last null value
            for (int r = 0; r < size; r++) {
                float value = list.get(r).values[index];
                if (value == Float.NEGATIVE_INFINITY) {
                    if (first == -1) {
                        first = r;
                    }
                    last = r;
                }
                if (first != -1
                        && (value != Float.NEGATIVE_INFINITY || r == size - 1)) {
                    // fill interval [first, last] of null values
                    // with average value(first - 1) + value(last + 1) / 2
                    float firstValue = first > 0 ? list.get(first - 1).values[index] : Float.NEGATIVE_INFINITY;
                    float lastValue = last < size - 1 ?
                            list.get(last + 1).values[index] : Float.NEGATIVE_INFINITY;
                    // set first = last or vice-versa if one is -1
                    firstValue = firstValue != Float.NEGATIVE_INFINITY ? firstValue : lastValue;
                    lastValue = lastValue != Float.NEGATIVE_INFINITY ? lastValue : firstValue;
                    // if fillValue == -1, then all the list is null
                    float fillValue = (firstValue + lastValue) / 2;
                    for (int n = first; n <= last; n++) {
                        list.get(n).fill(index, fillValue);
                    }
                }
            }
        }
        return list;
    }

    /**
     * Samples the list based on given time-type
     * @param list
     * @return
     */
    public static ArrayList<BJRow> sampleTime(ArrayList<BJRow> list, String mode){
        Function<String, String> getGroup = new MyDateFormat(mode).getGrouper();
        // Aggregate / Sample the time
        HashMap<String, BJRow> aggregate = new HashMap<>(list.size());
        for(BJRow record : list){
            String group = getGroup.apply(record.utcTime);
            BJRow groupAggregate = aggregate.get(group);
            if(groupAggregate == null){
                record.utcTime = group; // the representative of group time
                aggregate.put(group, record); // first record of this group
                continue;
            }
            // sum the group values and increase the aggregation count
            float[] aggValues = groupAggregate.values;
            for(int v = 0; v < aggValues.length ; v++){
                aggValues[v] += record.values[v];
            }
            groupAggregate.count++;
        }
        ArrayList<BJRow> samples = new ArrayList<>(aggregate.values());
        // sort aggregated samples by time
        Collections.sort(samples, Comparator.comparing(r1 -> r1.utcTime));
        // Change aggregate sum to average using the recorded group size
        for(BJRow sample : samples){
            for(int v = 0 ; v < sample.values.length ; v++){
                sample.values[v] /= sample.count;
            }
        }
        return samples;
    }

    /**
     * Convert the given column names and their corresponding values
     * to BJRow object
     * @param columns column names
     * @param values corresponding values per column
     * @return
     */
    @Override
    public BJRow map(String [] columns, String[] values){
        BJRow row = new BJRow();
        for(int c = 0 ; c < columns.length ; c++){
            String column = columns[c];
            String value = values[c];
            switch (column) {
                case STATION_ID:
                case STATION_ID2:
                    row.stationId = value
                            .replace("_meo", "")
                            .replace("_aq", "");
                    break;
                case STATION_TYPE:
                    row.stationType = value;
                    break;
                case TIME:
                    row.utcTime = value;
                    break;
                case WEATHER:
                    row.weather = value;
                    break;
                default:
                    int index = index(column);
                    if(index >= 0) row.values[index] = Util.toFloat(value);
                    break;
            }
        }
        return row;
    }

    /**
     * Map data-set of Row to data-set of KDDRow
     * @param dataSet
     * @return
     */
    public Dataset<BJRow> map(Dataset<Row> dataSet){
        final String[] columns = dataSet.columns();
        return dataSet.map(
                (MapFunction<Row, BJRow>) value -> (BJRow) map(columns, value)
                , Encoders.bean(BJRow.class));
    }

    /**
     * Return float value based on its column name
     * @param columnName
     * @return
     */
    public float get(String columnName){
        return values[index(columnName)];
    }


    public void setStationId(String stationId) {
        this.stationId = stationId;
    }

    public void setStationType(String stationType) {
        this.stationType = stationType;
    }

    public void setUtcTime(String utcTime) {
        this.utcTime = utcTime;
    }

    public void setWeather(String weather) {
        this.weather = weather;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public void setValues(float[] values) {
        this.values = values;
    }

    public void setFilled(boolean[] filled) {
        this.filled = filled;
    }

    public String getStationId() {
        return stationId;
    }

    public String getStationType() {
        return stationType;
    }

    public String getUtcTime() {
        return utcTime;
    }

    public String getWeather() {
        return weather;
    }

    public int getCount() {
        return count;
    }

    public float[] getValues() {
        return values;
    }

    public boolean[] getFilled() {
        return filled;
    }
}
