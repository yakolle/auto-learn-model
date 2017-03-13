package org.automl.model.utils

/**
  * Created by zhangyikuo on 2016/9/6.
  */
object SimilarityUtil {
  /**
    * 计算欧式距离，并按照相应权重进行加权
    *
    * @param x1      向量1
    * @param x2      向量2
    * @param weights 权重
    * @return 加权后的欧式距离
    */
  def calcWeightedEuclideanDistance(x1: Array[Double], x2: Array[Double], weights: Array[Double]): Double = {
    var sum = 0.0
    for (i <- weights.indices) {
      val dist = x1(i) - x2(i)
      sum += weights(i) * weights(i) * dist * dist
    }
    math.sqrt(sum)
  }

  /**
    * 计算欧式距离
    *
    * @param x1 向量1
    * @param x2 向量2
    * @return 欧式距离
    */
  def calcEuclideanDistance(x1: Array[Double], x2: Array[Double]): Double = {
    var sum = 0.0
    for (i <- x1.indices) {
      val dist = x1(i) - x2(i)
      sum += dist * dist
    }
    math.sqrt(sum)
  }
}
