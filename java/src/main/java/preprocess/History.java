package preprocess;
import utils.Util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class History {

    public HashMap<String, Boolean> isNotValue; // if the column holds value
    public HashMap<String, Integer> index; // maps column name to index
    public String[] columns; // name of column names
    public String[][] strings; // matrix of string data
    public float[][] values; // matrix of numeric data
    public boolean[][] missing; // indicator of missing values

    /**
     * Indicate those columns that are not value to be placed in matrix of strings
     * @param nonValueColumns
     */
    public History(String[] nonValueColumns){
        isNotValue = new HashMap<>(nonValueColumns.length);
        for(String nonValueColumn : nonValueColumns) {
            isNotValue.put(nonValueColumn, true);
        }
    }

    /**
     * Initialize History structure from given string array of rows
     * First row must contain the column names
     * @param data
     * @return
     */
    public History init(ArrayList<String[]> data){
        columns = data.get(0);
        index = new HashMap<>(columns.length);
        boolean isValue[] = new boolean[columns.length]; // fast access to value indicator
        int indices[] = new int[columns.length]; // fast access to indices
        int nonValueColumn = 0; // index of a string column
        int valueColumn = 0; // index of a numeric column
        for(int c = 0 ; c < columns.length ; c++){
            isValue[c] = isValue(columns[c]);
            if(isValue[c]){
                index.put(columns[c], valueColumn);
                indices[c] = valueColumn++;
            }else{
                index.put(columns[c], nonValueColumn);
                indices[c] = nonValueColumn++;
            }
        }
        // Initialize the arrays (reminder: first row of data is column names)
        strings = new String[data.size() - 1][nonValueColumn];
        values = new float[data.size() - 1][valueColumn];
        missing = new boolean[data.size() - 1][index.size()];
        for(int r = 0 ; r < data.size() - 1 ; r++){
            String[] column = data.get(r + 1);
            for(int c = 0 ; c < column.length ; c++){
                if(isValue[c]){
                    float value = Util.toFloat(column[c]);
                    values[r][indices[c]] = value;
                    missing[r][c] = value == Float.NEGATIVE_INFINITY;
                }else{
                    strings[r][indices[c]] = column[c];
                    missing[r][c] = column[c].length() == 0;
                }
            }
        }
        return this;
    }


    /**
     * Fill in missing (null) values with the average of nearest neighbors
     * and missing strings with nearest previous neighbor
     * @return
     */
    public History fill(){
        for(int c = 0 ; c < columns.length ; c++) {
            if (isValue(columns[c])){
                fillValue(index(columns[c]));
            }else{

            }
        }
        return this;
    }

    /**
     * Fill the missing values of given column index for a numeric column
     * Set the first non-nan value or the last if the data is started with NaN values
     * @param columnIndex
     * @return
     */
    public History fillValue(int columnIndex){
        int first = -1; // first null value
        int last = -1; // last null value
        for (int r = 0; r < strings.length; r++) {
            String value = strings[r][columnIndex];
            if (value.length() == 0) {
                if (first == -1) {
                    first = r;
                }
                last = r;
            }
            if (first != -1
                    && (value.length() > 0 || r == strings.length - 1)) {
                // fill interval [first, last] of null values
                // with average value(first - 1) + value(last + 1) / 2
                float firstValue = first > 0 ?
                        values[first - 1][columnIndex] : Float.NEGATIVE_INFINITY;
                float lastValue = last < values.length - 1 ?
                        values[last + 1][columnIndex] : Float.NEGATIVE_INFINITY;
                // set first = last or vice-versa if one is -1
                firstValue = firstValue != Float.NEGATIVE_INFINITY ? firstValue : lastValue;
                lastValue = lastValue != Float.NEGATIVE_INFINITY ? lastValue : firstValue;
                // if fillValue == -1, then all the list is null
                float fillValue = (firstValue + lastValue) / 2;
                for (int n = first; n <= last; n++) {
                    values[r][n] = fillValue;
                }
            }
        }
        return this;
    }

    /**
     * Fill the missing values of given column index for a string column
     * @param columnIndex
     * @return
     */
    public History fillString(int columnIndex){
        int first = -1; // first null value
        int last = -1; // last null value
        for (int r = 0; r < values.length; r++) {
            float value = values[r][columnIndex];
            if (value == Float.NEGATIVE_INFINITY) {
                if (first == -1) {
                    first = r;
                }
                last = r;
            }
            if (first != -1
                    && (value != Float.NEGATIVE_INFINITY || r == values.length - 1)) {
                // fill interval [first, last] of null values
                // with average value(first - 1) + value(last + 1) / 2
                float firstValue = first > 0 ?
                        values[first - 1][columnIndex] : Float.NEGATIVE_INFINITY;
                float lastValue = last < values.length - 1 ?
                        values[last + 1][columnIndex] : Float.NEGATIVE_INFINITY;
                // set first = last or vice-versa if one is -1
                firstValue = firstValue != Float.NEGATIVE_INFINITY ? firstValue : lastValue;
                lastValue = lastValue != Float.NEGATIVE_INFINITY ? lastValue : firstValue;
                // if fillValue == -1, then all the list is null
                float fillValue = (firstValue + lastValue) / 2;
                for (int n = first; n <= last; n++) {
                    values[r][n] = fillValue;
                }
            }
        }
        return this;
    }

    /**
     * Column index of given name
     * @param columnName
     * @return
     */
    public int index(String columnName){
        return index.get(columnName);
    }

    public boolean isValue(String columnName){
        return !isNotValue.containsKey(columnName);
    }

    /**
     * Convert data to CSV compatible format
     * lines of delimited values with header
     * @return
     */
    public List<String> toCSV(){
        ArrayList<String> csv = new ArrayList<>(values.length);
        for(int r = 0; r < columns.length ; r++){

        }
        return csv;
    }
}
