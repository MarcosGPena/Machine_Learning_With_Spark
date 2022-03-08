package com.example;

import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;



public class DecisionTree {
    
    public static void main(String[] args) {


        SparkSession spark = SparkSession.builder().master("local").getOrCreate();
        
        Dataset<Row> df = spark.read().format("csv")
        .option("header","true")
        .option("inferSchema", true)
        .load(".../wine_dataset.csv");
        
        df.show(2);
       

     VectorAssembler assembler = new VectorAssembler()
     .setInputCols(new String[] {"fixed_acidity","chlorides","pH"})
     .setOutputCol("features");
        
     StringIndexer styleIndexer = new StringIndexer();

     styleIndexer.setInputCol("style");
     styleIndexer.setOutputCol("style_2");

     df = styleIndexer.fit(df).transform(df);

     new IndexToString()
     .setInputCol("style_2")
     .setOutputCol("value")
     .transform(df.select("style_2").distinct());
    
        
     Dataset<Row> dadosFeatures = assembler.transform(df).select("style_2","features").withColumnRenamed("style_2", "label");


    DecisionTreeClassifier dTreeClassifier = new DecisionTreeClassifier();

    
    Dataset<Row>[] trainTestData = dadosFeatures.randomSplit(new double[] {0.8,0.2});
    Dataset<Row> trainData = trainTestData[0];
    Dataset<Row> testData = trainTestData[1];

    DecisionTreeClassificationModel model_1 = dTreeClassifier.fit(trainData);
    Dataset<Row> predicoes = model_1.transform(testData);

    predicoes.show();

    MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
    evaluator.setMetricName("accuracy");

    System.out.println("Accuracy : " + evaluator.evaluate(predicoes));
    
       spark.close();
        
    }

    
}
