package org.automl.model.strategy.scheduler

import org.apache.spark.sql.DataFrame
import org.automl.model.strategy.ProbeTask
import org.automl.model.strategy.learn.LearnerBase
import org.automl.model.utils.SampleUtil

import scala.util.Random

/**
  * Created by zhangyikuo on 2016/8/19.
  */
abstract class ProbeSchedulerBase {
  protected val randomGenerator = Random
  protected val propagationRandomGenerator = Random
  protected val paramChooseRandomGenerator = Random

  protected var learner: LearnerBase = _

  //对已经探测过的线进行轮盘选择时，上下浮动的最大比率，当然是按照每条线的实际验证值进行轮盘选择
  var maxPeerFluctuationRatio = 0.1
  //本策略进行评估时，最大的评估容忍限度比率，越大容忍度越大
  protected var maxEstimateAcceptRatio = 0.2

  //如果策略选取的参数评估一直不通过，最大的尝试次数
  protected val maxSinkingTimes = 10

  def setMaxEstimateAcceptRatio(maxEstimateAcceptRatio: Double) {
    this.maxEstimateAcceptRatio = maxEstimateAcceptRatio
  }

  def getMaxEstimateAcceptRatio: Double = maxEstimateAcceptRatio

  def setLearner(learner: LearnerBase) {
    this.learner = learner
  }

  /**
    * 增大系统扰动
    */
  def amplifyFluctuation() {
    maxEstimateAcceptRatio = math.min(1.0, maxEstimateAcceptRatio + (1 - maxEstimateAcceptRatio) / maxSinkingTimes)
  }

  /**
    * 根据学习过程中的超参数数据进行参数评估模型的学习
    *
    * @param paramData 学习过程中的超参数数据
    */
  def learn(paramData: DataFrame) {
    learner.learn(paramData)
  }

  /**
    * 根据超参数数组进行超参数评估模型的在线学习
    *
    * @param paramArray 超参数数组，包括目标评估值
    */
  def onlineLearn(paramArray: Array[Double]) {
    learner.onlineLearn(paramArray)
  }

  /**
    * 根据历史超参数的实际评估值抽样选择要进行搜索的原型（即基于此做扩展搜索）
    *
    * @param paramMatrix 超参数数据
    * @return 进行搜索的原型的索引
    */
  protected def choosePropagationLine(paramMatrix: Array[Array[Double]]): Int = {
    var minValidation = paramMatrix.minBy(_.last).last
    minValidation -= 1E-6 * minValidation
    val peerWeights = paramMatrix.map(params => SampleUtil.getNextNonNegativeTrimmedGaussian(propagationRandomGenerator,
      params.last - minValidation, maxPeerFluctuationRatio / 3))
    SampleUtil.rouletteLikeSelect(propagationRandomGenerator, peerWeights)
  }

  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的超参数列表
    */
  def getNextParamsInternal(currentTask: ProbeTask): Array[Double]

  /**
    * 模板方法，子类如无必要，可不用重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的超参数列表
    */
  def getNextParams(currentTask: ProbeTask): Array[Double] = {
    val nextParams = getNextParamsInternal(currentTask)
    if (null == nextParams) null else currentTask.formatParams(nextParams)
  }

  /**
    * 获取下次要probe的任务，主要是获取新的要probe的超参数，该方法为框架性方法，是提供给外部获取任务的接口，子类无需重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的任务
    */
  def getNextProbeTask(currentTask: ProbeTask): ProbeTask = {
    //任务运行点置0
    currentTask.runPoint = 0

    val currentParams = currentTask.getParams
    val currentEstimate = currentTask.getFinalValidation

    var nextParams = currentParams
    var nextEstimate = 0.0
    var maxEstimateAcceptRatioNice = maxEstimateAcceptRatio
    var sinkingTimes = 0
    do {
      nextParams = getNextParams(currentTask)
      nextEstimate = learner.predict(nextParams)
      /*
      如果评估值大于当前任务的验证值，或者虽然小于但仍然在容许的限度范围内，就认为新生成的参数集合是可探测的,
      并且如果策略生成的参数评估多次失败，就线性提升容忍度（临时）
       */
      maxEstimateAcceptRatioNice = maxEstimateAcceptRatio + (1 - maxEstimateAcceptRatio) * sinkingTimes / maxSinkingTimes
      sinkingTimes += 1
    } while (nextEstimate < currentEstimate && 1 - nextEstimate / currentEstimate > maxEstimateAcceptRatioNice * randomGenerator.nextDouble)

    currentTask.updateParams(nextParams)
    currentTask
  }
}
