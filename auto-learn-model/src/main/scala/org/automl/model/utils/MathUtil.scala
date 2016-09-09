package org.automl.model.utils

/**
  * Created by zhangyikuo on 2016/9/7.
  */
object MathUtil {
  /**
    * 按照给定权重，计算2范长度
    *
    * @param x       向量
    * @param weights 权重
    * @return 2范长度
    */
  def calcWeightedNorm(x: Array[Double], weights: Array[Double]): Double = {
    var sum = 0.0
    for (i <- weights.indices) sum += weights(i) * weights(i) * x(i) * x(i)
    math.sqrt(sum)
  }

  /**
    * 计算2范长度
    *
    * @param x 向量
    * @return 2范长度
    */
  def calcNorm(x: Array[Double]): Double = {
    val norm = x.fold(0.0) {
      case (normSum, xIt) =>
        normSum + xIt * xIt
    }
    math.sqrt(norm)
  }

  /**
    * 计算数据平均值
    *
    * @param data 数据
    * @return 各列平均值数组
    */
  def calcMean(data: Array[Array[Double]]): Array[Double] = {
    val sum = Array.fill(data.head.length)(0.0)
    data.foreach(x => for (i <- x.indices) sum(i) += x(i))
    val rowNum = data.length
    sum.map(_ / rowNum)
  }

  /**
    * 计算两个向量点积
    *
    * @param x1 向量1
    * @param x2 向量2
    * @return 点积
    */
  def dot(x1: Array[Double], x2: Array[Double]): Double = {
    var sum = 0.0
    for (i <- x1.indices) sum += x1(i) * x2(i)
    sum
  }
}
