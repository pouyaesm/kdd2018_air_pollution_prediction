package utils;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.TimeZone;
import java.util.function.Function;

public class MyDateFormat {

    /**
     * Sample / Aggregate records of a list per day
     */
    public final static String DAY = "DAY";
    public final static String HOUR6 = "H6";

    private String mode;
    private SimpleDateFormat format;
    private Calendar calendar;

    /**
     * @param coarseGrainMode how to coarse-grain the time input
     */
    public MyDateFormat(String coarseGrainMode){
        this.mode = coarseGrainMode;
        String pattern = "";
        switch (mode){
            case DAY:
                pattern = "yyyy-MM-dd";
                break;
            case HOUR6:
                pattern = "yyyy-MM-dd HH:00:00";

        }
        format = new SimpleDateFormat(pattern);
        format.setTimeZone(TimeZone.getTimeZone("GMT"));
        calendar = Calendar.getInstance();
        calendar.setTimeZone(TimeZone.getTimeZone("GMT"));
    }

    /**
     * Return the function that receives a string date and generates
     * A date that is coarse-grained based on mode
     * @return
     */
    public Function<String, String> getGrouper(){
        switch (mode){
            case DAY:
                return (String dateTime) -> {
                    try {
                        return format.format(format.parse(dateTime)) + " 00:00:00";
                    } catch (ParseException e) {
                        return "";
                    }
                };
            case HOUR6:
                return (String time) -> {
                    try {
                        calendar.setTime(format.parse(time));
                        // map [21, 24] to hour = 0
                        int hour = calendar.get(Calendar.HOUR_OF_DAY) % 21;
                        hour = (int)(6 * Math.round(hour / 6.0)); // round hour to 6K
                        calendar.set(Calendar.HOUR_OF_DAY, hour);
                        return format.format(calendar.getTime());
                    } catch (ParseException e) {
                        return "";
                    }
                };
        }
        return null;
    }
}
