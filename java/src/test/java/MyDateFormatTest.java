import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import utils.MyDateFormat;
import java.util.Arrays;
import java.util.Collection;

@RunWith(Parameterized.class)
public class MyDateFormatTest {

    private String mode;
    private String input;
    private String expected;

    public MyDateFormatTest(String mode, String input, String expected){
        this.mode = mode;
        this.input = input;
        this.expected = expected;
    }

    @Test
    public void test() {
        MyDateFormat dateFormat = new MyDateFormat(mode);
        String date = dateFormat.getGrouper().apply(input);
        Assert.assertEquals(expected, date);
    }

    // Provide data
    @Parameterized.Parameters
    public static Collection data() {
        return Arrays.asList(new String[][] {
                // Round to day
                {MyDateFormat.DAY, "2018-01-01 12:00:00", "2018-01-01 00:00:00"},
                // Round to 6 hours
                {MyDateFormat.HOUR6, "2018-01-01 2:00:00", "2018-01-01 00:00:00"},
                {MyDateFormat.HOUR6, "2018-01-01 10:00:00", "2018-01-01 12:00:00"},
                {MyDateFormat.HOUR6, "2018-01-01 17:00:00", "2018-01-01 18:00:00"},
                {MyDateFormat.HOUR6, "2018-01-01 23:00:00", "2018-01-01 00:00:00"},
        });
    }
}
