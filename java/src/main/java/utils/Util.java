package utils;

import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.io.Files;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.StructType;
import scala.collection.JavaConversions;
import scala.collection.Seq;

import java.io.*;
import java.net.URL;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

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

    /**
     * Read a cell separated data (e.g., CSV) with[out] a header
     * @param address
     * @param delimiter
     * @return
     */
    public static ArrayList<String[]> read(String address, String delimiter){
        ArrayList<String[]> list = new ArrayList<>();
        try(BufferedReader br = new BufferedReader(new FileReader(address))) {
            String line;
            while ((line = br.readLine()) != null) {
                // limit = -1 to keep empty values at the end of line
                list.add(line.split(delimiter, -1));
            }
        }catch (IOException e) {
            e.printStackTrace();
        }
        return list;
    }

    /**
     * Read csv into Spark DataSet
     * @param address
     * @return
     */
    public static Dataset<Row> read(String address, StructType schema){
        SparkSession sparkSession = SparkSetup.getSession();
        DataFrameReader dataFrameReader = sparkSession.read()
                .option("mode", "PERMISSIVE")
                .option("header", "true");
        if(schema == null){
            dataFrameReader.option("inferSchema", "true");
        }else{
            dataFrameReader.schema(schema);
        }
        return dataFrameReader.csv(address);
    }

    /**
     * Read a cell separated data from URL
     * @param address
     * @param separator
     * @return
     */
    public static ArrayList<String[]> readURL(String address, String separator){
        ArrayList<String[]> list = new ArrayList<>();
        try {
            URL url = new URL(address);
            Scanner scn = new Scanner(url.openStream());
            while(scn.hasNextLine()){
                // limit = -1 to keep empty values at the end of line
                list.add(scn.nextLine().split(separator, -1));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return list;
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

    public static int toInt(Object value){
        try{
            return Integer.parseInt(value.toString());
        }catch (Exception exp){
            return Integer.MIN_VALUE;
        }
    }

    public static String toString(Object value){
        try{
            return value.toString();
        }catch (Exception exp){
            return "";
        }
    }

    /**
     * Convert array of values to Scala seq
     * @param values
     * @return
     */
    public static Seq<String> toSeq(String... values) {
        return JavaConversions.asScalaBuffer(Lists.newArrayList(values));
    }

    /**
     * Fast write of array of strings to a file
     * @param records
     * @param address
     */
    private static void write(List<String> records, String address) {
        int bufferSize = 16777216 ; // 16M
        File file = new File(address);
        try {
            FileWriter writer = new FileWriter(file);
            BufferedWriter bufferedWriter = new BufferedWriter(writer, bufferSize);
            for (String record: records) {
                bufferedWriter.write(record);
            }
            bufferedWriter.flush();
            bufferedWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static double SMAPE(){
        return 0;
    }
}
