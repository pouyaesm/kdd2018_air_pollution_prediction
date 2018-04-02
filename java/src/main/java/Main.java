
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import preprocess.KDDDataSet;
import utils.Config;

public class Main {
    public static void main(String[] args) {
        KDDDataSet dataSet = new KDDDataSet();
        Dataset<Row> records = dataSet.load(Config.get(Config.CLEAN_DATA)).get();
        records.show(5);
    }
}
