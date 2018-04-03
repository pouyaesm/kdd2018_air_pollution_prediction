
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import preprocess.ObservedData;
import utils.Config;

public class Main {
    public static void main(String[] args) {
        ObservedData dataSet = new ObservedData();
        Dataset<Row> records = dataSet.load(Config.get(Config.BEIJING_OBSERVED)).get();
        records.show(5);
    }
}
