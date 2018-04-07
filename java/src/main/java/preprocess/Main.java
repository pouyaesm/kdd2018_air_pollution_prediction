package preprocess;

import utils.Config;
import utils.Util;

import java.util.ArrayList;

public class Main {
    public static void main(String[] args) {
        ObsHistory data = new ObsHistory();
        ArrayList<String[]> list = Util.read(Config.get(Config.BEIJING_OBS), ";");
        data.init(list);
        System.out.println("helloo");
    }
}
