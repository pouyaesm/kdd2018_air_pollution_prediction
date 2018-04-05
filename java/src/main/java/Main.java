
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import preprocess.BJFiles;
import utils.Config;

public class Main {
    public static void main(String[] args) {
        BJFiles dataSet = new BJFiles();
        Dataset<Row> records = dataSet.load(Config.get(Config.BEIJING_OBSERVED)).get();
        records.show(5);
    }
}
