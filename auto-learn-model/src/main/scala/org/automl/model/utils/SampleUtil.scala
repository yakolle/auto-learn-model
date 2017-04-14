package org.automl.model.utils

import scala.util.Random

/**
  * Created by zhangyikuo on 2016/8/26.
  */
object SampleUtil {
  /**
    * 计算cdf
    *
    * @param pArray 离散pdf数组
    * @return 离散pdf数组
    */
  def cdf(pArray: Array[Double]): Array[Double] = {
    var tmpSum = 0.0
    for (p <- pArray) yield {
      tmpSum += p
      tmpSum
    }
  }

  /**
    * 轮盘赌选择法，实际上就是按照指定概率分布进行抽样
    *
    * @param randomGenerator 随机数产生器
    * @param pArray          各轮盘大小
    * @return 被选中的轮盘编号
    */
  def rouletteLikeSelect(randomGenerator: Random, pArray: Array[Double]): Int = rouletteLikeSelect(randomGenerator, pArray, 1).head

  /**
    * 轮盘赌选择法，实际上就是按照指定概率分布进行抽样
    *
    * @param randomGenerator 随机数产生器
    * @param pArray          各轮盘大小
    * @param num             要抽样的个数
    * @return 被选中的轮盘编号数组
    */
  def rouletteLikeSelect(randomGenerator: Random, pArray: Array[Double], num: Int): Array[Int] = {
    val indices = new Array[Int](num)
    val cdfArray = cdf(pArray)

    for (i <- 0 until num) {
      val acceptRatio = randomGenerator.nextDouble * cdfArray.last
      indices(i) = cdfArray.indexWhere(_ >= acceptRatio)
    }

    indices
  }

  /**
    * 获取高斯随机数，并且以mean为期望，stdWidthRatio*mean为标准差
    *
    * @param randomGenerator 随机数产生器
    * @param mean            期望
    * @param stdWidthRatio   标准差宽度占比，标准差与期望的比率
    * @return 经过处理后的高斯随机数
    */
  def getNextGaussian(randomGenerator: Random, mean: Double, stdWidthRatio: Double = 0.1): Double =
    mean + randomGenerator.nextGaussian * stdWidthRatio * math.abs(mean)

  /**
    * 获取高斯随机数，以mean为期望，stdWidthRatio*mean为标准差，并且以stdWidthRatio*mean*thresholdRatio为界限截断尾部
    *
    * @param randomGenerator 随机数产生器
    * @param mean            期望
    * @param stdWidthRatio   标准差宽度占比，标准差与期望的比率
    * @param thresholdRatio  对thresholdRatio个标准差后的高斯数进行截断
    * @return 经过处理后的高斯随机数
    */
  def getNextTrimmedGaussian(randomGenerator: Random, mean: Double, stdWidthRatio: Double = 0.1, thresholdRatio: Double = 3.0): Double = {
    val threshold = math.abs(mean) * stdWidthRatio * thresholdRatio
    val gaussian = getNextGaussian(randomGenerator, mean, stdWidthRatio)
    if (gaussian > mean + threshold) mean + threshold else if (gaussian < mean - threshold) mean - threshold else gaussian
  }

  /**
    * 获取非负高斯随机数，以mean为期望，stdWidthRatio*mean为标准差，并且以stdWidthRatio*mean*thresholdRatio为界限截断尾部
    *
    * @param randomGenerator 随机数产生器
    * @param mean            期望
    * @param stdWidthRatio   标准差宽度占比，标准差与期望的比率
    * @param thresholdRatio  对thresholdRatio个标准差后的高斯数进行截断
    * @return 经过处理后的高斯随机数
    */
  def getNextNonNegativeTrimmedGaussian(randomGenerator: Random, mean: Double, stdWidthRatio: Double = 0.1, thresholdRatio: Double = 3.0): Double = {
    val gaussian = getNextTrimmedGaussian(randomGenerator, mean, stdWidthRatio, thresholdRatio)
    if (gaussian < 0.0) 0.0 else gaussian
  }
}
