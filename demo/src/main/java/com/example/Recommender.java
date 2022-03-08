package com.example;

import org.apache.spark.sql.Row;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;



public class Recommender {

    public static void main(String[] args) {
        SparkSession spark = SparkSession.builder().master("local").getOrCreate();
        
      
       Dataset<Row> ratings = spark.read().format("csv")
        .option("header","true")
        .option("inferSchema", true)
        .load(".../Books_Recommendation/BX-CSV-Dump/Book-Ratings.csv");
        
        for(String c : ratings.columns()){
            ratings = ratings.withColumn(c, ratings.col(c).cast(DataTypes.IntegerType)).na().drop();
        }
       
        
        Dataset<Row>[] splits = ratings.randomSplit(new double[]{0.8,0.2});
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        
        ALS als = new ALS()
        .setMaxIter(5)
        .setRegParam(0.01)
        .setImplicitPrefs(true)
        .setUserCol("userId")
        .setItemCol("ISBN")
        .setRatingCol("bookRating");

        ALSModel model = als.fit(train);
        model.setColdStartStrategy("drop");
        Dataset<Row> predictions = model.transform(test);

        Dataset<Row> users = ratings.select(als.getUserCol()).distinct().limit(3);
        Dataset<Row> userSubsetRecs = model.recommendForUserSubset(users, 5);

        userSubsetRecs.show(false);

       
        spark.close();

        }
    
}
