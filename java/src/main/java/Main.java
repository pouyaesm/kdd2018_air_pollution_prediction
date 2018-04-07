
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import depricated.BJFiles;
import utils.Config;

public class Main {
    public static void main(String[] args) {
        BJFiles dataSet = new BJFiles();
        Dataset<Row> records = dataSet.load(Config.get(Config.BEIJING_OBS)).get();
        records.show(5);
    }
}
