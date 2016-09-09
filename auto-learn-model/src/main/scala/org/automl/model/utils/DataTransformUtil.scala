package org.automl.model.utils

import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.feature.{VectorAssembler, VectorSlicer}
import org.apache.spark.sql.DataFrame

/**
  * Created by zhangyikuo on 2016/9/30.
  */
object DataTransformUtil {
  /**
    * 将数据转换为向量形式（所有的输入X汇集成向量形式）
    *
    * @param data 原数据（每一列为一个特征）
    * @return 新数据（只有两列，一列为features——原来输入数据X汇集的向量形式，一列为target——原来的target列）
    */
  def dataSchemaTransform(data: DataFrame) = new VectorAssembler().setInputCols(data.columns.filter(_ != "label"))
    .setOutputCol("features").transform(data).select("features", "label")

  /**
    * 根据提供的列名，选出这些列。如果selectedColNames的长度小于等于1，不进行处理
    *
    * @param data             原数据
    * @param selectedColNames 需要选出的列名
    * @return selectedColNames对应列的数据
    */
  def selectColumnByNames(data: DataFrame, selectedColNames: Array[String]): DataFrame = {
    if (selectedColNames.length <= 1) data
    else {
      val firstColName = selectedColNames.head
      val otherColNames = selectedColNames.tail
      data.select(firstColName, otherColNames: _*)
    }
  }

  /**
    * 从Vector类型的features中选出指定的features
    *
    * @param data                 原数据
    * @param selectedFeatureNames 需要选出的features
    * @return selectedFeatureNames对应的features
    */
  def selectFeaturesFromAssembledData(data: DataFrame, selectedFeatureNames: Array[String]): DataFrame = new VectorSlicer()
    .setInputCol("features").setNames(selectedFeatureNames).transform(data)

  /**
    * 从Vector类型数据中获取features的name
    *
    * @param data 原数据
    * @return 数据features的name
    */
  def extractFeatureNamesFromAssembledData(data: DataFrame): Array[String] = AttributeGroup.fromStructField(data.schema("features"))
    .attributes.get.map(_.name.get)
}
