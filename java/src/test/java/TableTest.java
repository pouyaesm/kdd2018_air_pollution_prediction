import org.junit.Assert;
import org.junit.Test;
import utils.Util;

import java.util.ArrayList;

public class TableTest {
    @Test
    public void readTest(){
        ArrayList<String[]> list = Util.read(
                "src\\test\\data\\table1.csv", ",");
        Assert.assertArrayEquals(new String[] {"name", "sid", "height", "age"}, list.get(0));
        Assert.assertArrayEquals(new String[]{"ali", "", "180.2", ""}, list.get(1));
    }
}
