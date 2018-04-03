import org.junit.Assert;
import org.junit.Test;
import preprocess.ObservedRow;
import utils.MyDateFormat;

import java.util.ArrayList;

public class ObservedRowTest {

    @Test
    public void fillTest(){
        ArrayList<ObservedRow> list = new ArrayList<>();
        // [10, -inf, 30]
        list.add(new ObservedRow("1", "",
                10, Float.NEGATIVE_INFINITY, 30));
        // [-inf, 30, -inf]
        list.add(new ObservedRow("1", "",
                Float.NEGATIVE_INFINITY, 30, Float.NEGATIVE_INFINITY));
        // [30, 40, -inf]
        list.add(new ObservedRow("1", "",
                30, 40, Float.NEGATIVE_INFINITY));

        // fill PM2.5, PM10, O3
        ObservedRow.fill(list, new int[]{ObservedRow.PM25, ObservedRow.PM10, ObservedRow.O3});

        Assert.assertEquals(30, list.get(0).values[ObservedRow.PM10], 0);
        Assert.assertEquals(true, list.get(0).filled[ObservedRow.PM10]);
        Assert.assertEquals(20, list.get(1).values[ObservedRow.PM25], 0);
        Assert.assertEquals(30, list.get(1).values[ObservedRow.O3], 0);
        Assert.assertEquals(30, list.get(2).values[ObservedRow.O3], 0);
    }

    @Test
    public void sampleTimeDayTest(){
        ArrayList<ObservedRow> list = new ArrayList<>();
        list.add(new ObservedRow("1", "2018-01-01 12:00:00", 1, 2, 3));
        list.add(new ObservedRow("1", "2018-01-01 14:00:00", 2, 3, 4));
        ArrayList<ObservedRow> sample = ObservedRow.sampleTime(list, MyDateFormat.DAY);
        Assert.assertArrayEquals(new float[]{1.5f, 2.5f, 3.5f}, sample.get(0).values, 0);
    }

    @Test
    public void sampleTime6HoursTest(){
        ArrayList<ObservedRow> list = new ArrayList<>();
        list.add(new ObservedRow("1", "2018-01-01 3:00:00", 1, 2, 3));
        list.add(new ObservedRow("1", "2018-01-01 6:00:00", 2, 3, 4));
        list.add(new ObservedRow("1", "2018-01-01 15:00:00", 3, 4, 5));
        ArrayList<ObservedRow> sample = ObservedRow.sampleTime(list, MyDateFormat.HOUR6);
        Assert.assertArrayEquals(new float[]{1.5f, 2.5f, 3.5f}, sample.get(0).values, 0);
        Assert.assertEquals("2018-01-01 18:00:00", sample.get(1).utcTime);
    }
}
