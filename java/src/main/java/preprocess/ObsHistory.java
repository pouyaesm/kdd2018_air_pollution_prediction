package preprocess;

public class ObsHistory extends  History{
    // Column names in data-sets
    public final static String STATION_ID = "station_id";
    public final static String STATION_TYPE = "station_type";
    public final static String LONG = "longitude";
    public final static String LAT = "latitude";
    public final static String TIME = "utc_time";

    public final static String PM25 = "PM2.5";
    public final static String PM10 = "PM10";
    public final static String O3 = "O3";
    public final static String NO2 = "NO2";
    public final static String CO = "CO";
    public final static String SO2 = "SO2";

    public final static String TEMP = "temperature";
    public final static String PRES = "pressure";
    public final static String HUM = "humidity";
    public final static String W_DIR = "wind_direction";
    public final static String W_SPEED = "wind_speed";
    public final static String WEATHER = "weather";

    public ObsHistory(){
        super(new String[]{"station_id", "station_type", "utc_time", "weather"});
    }
}
