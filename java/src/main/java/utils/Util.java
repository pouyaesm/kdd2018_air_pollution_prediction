package utils;

import com.google.common.base.Charsets;
import com.google.common.io.Files;

import java.io.File;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.TimeZone;

public class Util {

    private static SimpleDateFormat simpleDateFormat;

    /**
     * Read a file text into string
     * @param address
     * @return
     */
    public static String read(String address){
        try {
            return Files.toString(new File(address), Charsets.UTF_8);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return "";
    }

    public static long toTime(String dateTime){
        Date date = toDate(dateTime);
        return date == null ? -1 : date.getTime() / 1000L;
    }

    public static Date toDate(String dateTime){
        if(simpleDateFormat == null){
            simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            simpleDateFormat.setTimeZone(TimeZone.getTimeZone("GMT"));
        }
        try {
            return simpleDateFormat.parse(dateTime);
        } catch (ParseException e) {
            return null;
        }
    }

    public static float toFloat(Object value){
        try{
            return Float.parseFloat(value.toString());
        }catch (Exception exp){
            return Float.NEGATIVE_INFINITY;
        }
    }

    public static double SMAPE(){
        return 0;
    }
}
