package utils;

import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.sql.SparkSession;

public class SparkSetup {

    private static SparkSession sparkSession;
    private static SparkContext sparkContext;

    public synchronized static String getCurrentPath(){
        return System.getProperty("user.dir");
    }

    public synchronized static void init(){
        System.setProperty("hadoop.home.dir", Config.get(Config.HADOOP));
    }

    public synchronized static SparkSession getSession(){
        init();
        if(sparkSession == null){
            sparkSession = SparkSession
                    .builder()
                    .appName("Java Spark Lab")
                    .master("local[4]")
                    .config("spark.sql.warehouse.dir", Config.get(Config.SPARK_WAREHOUSE))
                    .sparkContext(getContext())
                    .getOrCreate();
        }
        return sparkSession;
    }

    public synchronized static SparkContext getContext(){
        if(sparkContext == null){
            SparkConf sparkConf = new SparkConf()
                    .setAppName("SparkApp")
                    .setMaster("local[4]")
                    .set("spark.hadoop.validateOutputSpecs", "false") //to allow overwrites
                    .set("spark.local.dir", Config.get(Config.SPARK_TEMP));
            sparkContext = new SparkContext(sparkConf);
            sparkContext.setLogLevel("WARN");//only output warnings
        }
        return sparkContext;
    }
}
