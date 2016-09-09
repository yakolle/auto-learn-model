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
    * @param pArray 各轮盘大小
    * @return 被选中的轮盘编号
    */
  def rouletteLikeSelect(pArray: Array[Double]): Int = {
    val cdfArray = cdf(pArray)
    val acceptRatio = Random.nextDouble * cdfArray.last
    cdfArray.indexWhere(_ >= acceptRatio)
  }

  /**
    * 获取高斯随机数，并且以mean为期望，stdWidthRatio*mean为标准差
    *
    * @param randomGenerator 随机数产生器
    * @param mean            期望
    * @param stdWidthRatio   标准差宽度占比，标准差与期望的比率
    * @return 经过处理后的高斯随机数
    */
  def getNextGaussian(randomGenerator: Random, mean: Double, stdWidthRatio: Double = 0.1) =
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
  def getNextTrimmedGaussian(randomGenerator: Random, mean: Double, stdWidthRatio: Double = 0.1, thresholdRatio: Double = 3.0) = {
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
  def getNextNonNegativeTrimmedGaussian(randomGenerator: Random, mean: Double, stdWidthRatio: Double = 0.1, thresholdRatio: Double = 3.0) = {
    val gaussian = getNextTrimmedGaussian(randomGenerator, mean, stdWidthRatio, thresholdRatio)
    if (gaussian < 0.0) 0.0 else gaussian
  }
}
