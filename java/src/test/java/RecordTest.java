import org.junit.Assert;
import org.junit.Test;
import preprocess.Record;
import utils.MyDateFormat;
import utils.Util;

import java.util.ArrayList;

public class RecordTest {

    @Test
    public void fillTest(){
        ArrayList<Record> list = new ArrayList<>();
        // [10, -inf, 30]
        list.add(new Record("1", "",
                10, Float.NEGATIVE_INFINITY, 30));
        // [-inf, 30, -inf]
        list.add(new Record("1", "",
                Float.NEGATIVE_INFINITY, 30, Float.NEGATIVE_INFINITY));
        // [30, 40, -inf]
        list.add(new Record("1", "",
                30, 40, Float.NEGATIVE_INFINITY));

        // fill PM2.5, PM10, O3
        Record.fill(list, new int[]{Record.PM25, Record.PM10, Record.O3});

        Assert.assertEquals(30, list.get(0).values[Record.PM10], 0);
        Assert.assertEquals(20, list.get(1).values[Record.PM25], 0);
        Assert.assertEquals(30, list.get(1).values[Record.O3], 0);
        Assert.assertEquals(30, list.get(2).values[Record.O3], 0);
    }

    @Test
    public void sampleTimeDayTest(){
        ArrayList<Record> list = new ArrayList<>();
        list.add(new Record("1", "2018-01-01 12:00:00", 1, 2, 3));
        list.add(new Record("1", "2018-01-01 14:00:00", 2, 3, 4));
        ArrayList<Record> sample = Record.sampleTime(list, MyDateFormat.DAY);
        Assert.assertArrayEquals(new float[]{1.5f, 2.5f, 3.5f}, sample.get(0).values, 0);
    }

    @Test
    public void sampleTime6HoursTest(){
        ArrayList<Record> list = new ArrayList<>();
        list.add(new Record("1", "2018-01-01 3:00:00", 1, 2, 3));
        list.add(new Record("1", "2018-01-01 6:00:00", 2, 3, 4));
        list.add(new Record("1", "2018-01-01 15:00:00", 3, 4, 5));
        ArrayList<Record> sample = Record.sampleTime(list, MyDateFormat.HOUR6);
        Assert.assertArrayEquals(new float[]{1.5f, 2.5f, 3.5f}, sample.get(0).values, 0);
        Assert.assertEquals("2018-01-01 18:00:00", sample.get(1).utcTime);
    }
}
