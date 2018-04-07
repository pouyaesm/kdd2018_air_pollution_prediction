import org.junit.Assert;
import org.junit.Test;
import preprocess.History;

import java.util.ArrayList;

public class HistoryTest {
    @Test
    public void testHistoryConstruction(){
        History obsData = new History(new String[]{"station_id", "utc_time", "weather"});
        String[] columns = {"station_id", "utc_time", "O3", "weather"};
        String[] row1 = {"a", "12:00", "12.5", "sunny"};
        String[] row2 = {"b", "12:00", "", ""};
        ArrayList<String[]> data = new ArrayList<>();
        data.add(columns);
        data.add(row1);
        data.add(row2);
        obsData.init(data);
        // Numeric values
        Assert.assertArrayEquals(new float[]{12.5f}, obsData.values[0], 0);
        Assert.assertArrayEquals(new float[]{Float.NEGATIVE_INFINITY}, obsData.values[1], 0);
        // String values
        Assert.assertArrayEquals(new String[]{"a", "12:00", "sunny"}, obsData.strings[0]);
        Assert.assertArrayEquals(new String[]{"b", "12:00", ""}, obsData.strings[1]);
        // Missing values
        Assert.assertArrayEquals(new boolean[]{false, false, false, false}, obsData.missing[0]);
        Assert.assertArrayEquals(new boolean[]{false, false, true, true}, obsData.missing[1]);
    }
}
