import org.junit.Assert;
import org.junit.Test;
import utils.Util;

import java.lang.reflect.Array;
import java.util.ArrayList;

public class UtilTest {

    @Test
    public void toTimeTest() {
        Assert.assertEquals(1522652574L,
                Util.toTime("2018-04-02 07:02:54"));
    }

    @Test
    public void toFloatTest() {
        Assert.assertEquals(12.5f,
                Util.toFloat("12.5"), 0);
        Assert.assertEquals(Float.NEGATIVE_INFINITY,
                Util.toFloat(null), 0);
        Assert.assertEquals(Float.NEGATIVE_INFINITY,
                Util.toFloat("a12"), 0);
        Assert.assertEquals(Float.NEGATIVE_INFINITY,
                Util.toFloat(""), 0);
    }
}
