package preprocess;

import org.apache.spark.sql.Row;
import utils.MyDateFormat;
import utils.Util;

import java.io.Serializable;
import java.util.*;
import java.util.function.Function;

public class Record implements Serializable{
    public String stationId; // station stationId
    public String utcTime;
//    public long timestamp;
    public int count; // for aggregation purposes

    public float[] values = new float[3];

    public final static int PM25 = 0;
    public final static int PM10 = 1;
    public final static int O3 = 2;

    /**
     * Indicator of whether the missing value is filled
     */
    public boolean[] filled = new boolean[3];

    public Record(){

    }

    public Record(String stationId, String utcTime, float PM25, float PM10, float O3){
        init(stationId, utcTime, PM25, PM10, O3);
    }

    Record(Row row){
        init(row.getString(KDDDataSet.index(KDDDataSet.STATION_ID)),
                row.getString(KDDDataSet.index(KDDDataSet.TIME)),
                Util.toFloat(row.get(KDDDataSet.index(KDDDataSet.PM25))),
                Util.toFloat(row.get(KDDDataSet.index(KDDDataSet.PM10))),
                Util.toFloat(row.get(KDDDataSet.index(KDDDataSet.O3))));
    }

    private void init(String stationId, String utcTime, float PM25, float PM10, float O3){
        this.stationId = stationId;
        this.utcTime  = utcTime;
//        this.timestamp = Util.toTime(this.utcTime);
        this.values[Record.PM25] = PM25;
        this.values[Record.PM10] = PM10;
        this.values[Record.O3] = O3;
        this.count = 1;
    }

    /**
     * Fill the value and mark it as filled
     * @param index
     * @param value
     * @return
     */
    public Record fill(int index, float value){
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
    public static ArrayList<Record> fill(ArrayList<Record> list, int[] indices){
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
    public static ArrayList<Record> sampleTime(ArrayList<Record> list, String mode){
        Function<String, String> getGroup = new MyDateFormat(mode).getGrouper();
        // Aggregate / Sample the time
        HashMap<String, Record> aggregate = new HashMap<>(list.size());
        for(Record record : list){
            String group = getGroup.apply(record.utcTime);
            Record groupAggregate = aggregate.get(group);
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
        ArrayList<Record> samples = new ArrayList<>(aggregate.values());
        // sort aggregated samples by time
        Collections.sort(samples, Comparator.comparing(r1 -> r1.utcTime));
        // Change aggregate sum to average using the recorded group size
        for(Record sample : samples){
            for(int v = 0 ; v < sample.values.length ; v++){
                sample.values[v] /= sample.count;
            }
        }
        return samples;
    }
}
