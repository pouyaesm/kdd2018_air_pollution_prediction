import org.apache.spark.api.java.JavaPairRDD;
import org.junit.Assert;
import org.junit.Test;
import preprocess.ObservedData;
import preprocess.ObservedRow;
import utils.Util;

import java.util.ArrayList;

public class PreProcessTest {

    @Test
    public void testLoadSortGroup(){
        ObservedData data = new ObservedData();
        JavaPairRDD<String, ArrayList<ObservedRow>> stationData = data
                .load("src/test/data/beijing_17_18_sample.csv")
                .sort() // sort by time ascending
                .group().get(); // group each station into a list keyed by station stationId
        stationData.foreach(group -> {
            if(group._1.equals("xiayunling")){
                // Expect the 5th (last) record of xiayunling station
                // to have time = 2018-01-22 14:00:00 (column = 1)
                long timestamp = Util.toTime(group._2.get(4).utcTime);
                long expectedTime = Util.toTime("2018-01-22 14:00:00");
                Assert.assertEquals(expectedTime, timestamp);
            }
        });
    }
}
