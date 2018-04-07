import org.apache.spark.sql.Row;
import org.junit.Assert;
import org.junit.Test;
import depricated.BJRow;
import depricated.BJFiles;
import depricated.KDDRow;
import utils.Config;
import utils.MyDateFormat;

import java.util.ArrayList;

public class BJRowTest {

    @Test
    public void fillTest(){
        ArrayList<BJRow> list = new ArrayList<>();
        // [10, -inf, 30]
        list.add(new BJRow("1", "",
                10, Float.NEGATIVE_INFINITY, 30));
        // [-inf, 30, -inf]
        list.add(new BJRow("1", "",
                Float.NEGATIVE_INFINITY, 30, Float.NEGATIVE_INFINITY));
        // [30, 40, -inf]
        list.add(new BJRow("1", "",
                30, 40, Float.NEGATIVE_INFINITY));

        // fill PM2.5, PM10, O3
        BJRow.fill(list, new int[]{
                KDDRow.index(KDDRow.PM25), KDDRow.index(KDDRow.PM10), KDDRow.index(KDDRow.O3)});

        Assert.assertEquals(30, list.get(0).values[KDDRow.index(KDDRow.PM10)], 0);
        Assert.assertEquals(true, list.get(0).filled[KDDRow.index(KDDRow.PM10)]);
        Assert.assertEquals(20, list.get(1).values[KDDRow.index(KDDRow.PM25)], 0);
        Assert.assertEquals(30, list.get(1).values[KDDRow.index(KDDRow.O3)], 0);
        Assert.assertEquals(30, list.get(2).values[KDDRow.index(KDDRow.O3)], 0);
    }

    @Test
    public void sampleTimeDayTest(){
        ArrayList<BJRow> list = new ArrayList<>();
        list.add(new BJRow("1", "2018-01-01 12:00:00", 1, 2, 3));
        list.add(new BJRow("1", "2018-01-01 14:00:00", 2, 3, 4));
        ArrayList<BJRow> sample = BJRow.sampleTime(list, MyDateFormat.DAY);
        Assert.assertEquals(1.5f, sample.get(0).get(KDDRow.PM25), 0);
        Assert.assertEquals(2.5f, sample.get(0).get(KDDRow.PM10), 0);
        Assert.assertEquals(3.5f, sample.get(0).get(KDDRow.O3), 0);
    }

    @Test
    public void sampleTime6HoursTest(){
        ArrayList<BJRow> list = new ArrayList<>();
        list.add(new BJRow("1", "2018-01-01 3:00:00", 1, 2, 3));
        list.add(new BJRow("1", "2018-01-01 6:00:00", 2, 3, 4));
        list.add(new BJRow("1", "2018-01-01 15:00:00", 3, 4, 5));
        ArrayList<BJRow> sample = BJRow.sampleTime(list, MyDateFormat.HOUR6);
        Assert.assertEquals(1.5f, sample.get(0).get(KDDRow.PM25), 0);
        Assert.assertEquals(2.5f, sample.get(0).get(KDDRow.PM10), 0);
        Assert.assertEquals(3.5f, sample.get(0).get(KDDRow.O3), 0);
        Assert.assertEquals("2018-01-01 18:00:00", sample.get(1).utcTime);
    }

    @Test
    public void loadJoinSortTest(){
        String baseDir = "src\\test\\data\\";
        BJFiles bjFiles = new BJFiles()
                .set(Config.BEIJING_OBS_AQ, baseDir + "beijing_17_18_aq_sample.csv")
                .set(Config.BEIJING_OBS_AQ_REST, baseDir + "beijing_201802_201803_aq_sample.csv")
                .set(Config.BEIJING_OBS_MEO, baseDir + "beijing_17_18_meo_sample.csv")
                .set(Config.BEIJING_STATIONS, baseDir + "beijing_AirQuality_Stations_sample.csv")
                .load();
        String headers = String.join(",", bjFiles.getAll().columns());
        Row sample = bjFiles.getAll().first();
        Row[] stations = (Row[]) bjFiles.getStations().collect();
        Row[] all = (Row[]) bjFiles.getAll().collect();
        Assert.assertEquals("huairou,2018-01-30 16:00:00,4.0,34.0,4.0,0.4,81.0,5.0,116.626945,40.357777,-5.2,1022.8,27.0,30.0,0.8,Sunny/clear",
                sample.mkString(","));
        Assert.assertEquals("station_id,utc_time,PM2.5,PM10,NO2,CO,O3,SO2,longitude,latitude,temperature,pressure,humidity,wind_direction,wind_speed,weather",
                headers);
    }
}
