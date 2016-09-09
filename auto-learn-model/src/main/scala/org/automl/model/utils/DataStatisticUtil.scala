package org.automl.model.utils

import org.apache.spark.sql.DataFrame

/**
  * Created by zhangyikuo on 2017/1/17.
  */
object DataStatisticUtil {
  /**
    * 计算中位数
    *
    * @param data          数据，每列都为Double类型，而非Vector[Double]类型
    * @param relativeError 在实际中位数周围浮动的范围，比如data共N条数据，那么返回的中位数的浮动范围就是上下relativeError*N个
    * @return 中位数
    */
  def calcMedian(data: DataFrame, relativeError: Double = 0.01): Map[String, Double] = {
    data.columns.filter(_ != "label").map {
      colName =>
        (colName, data.stat.approxQuantile(colName, Array(0.5), relativeError).head)
    }.toMap
  }

  /**
    * 计算统计量
    *
    * @param data 数据，每列都为Double类型，而非Vector[Double]类型
    * @param op   操作类型，avg, max, min, sum, count
    * @return 统计量
    */
  def calcStatistic(data: DataFrame, op: String): Map[String, Double] = {
    val features = data.columns.filter(_ != "label")
    data.agg(features.map((_, op)).toMap).head().getValuesMap[Double](features)
  }
}
