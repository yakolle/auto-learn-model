package org.automl.model.strategy.scheduler

import java.util.concurrent.ConcurrentHashMap

import org.automl.model.context.ParamHoldler
import org.automl.model.strategy.ProbeTask
import org.automl.model.strategy.learn.LearnerBase
import org.automl.model.utils.SampleUtil

import scala.util.Random

/**
  * Created by zhangyikuo on 2016/8/23.
  */
class AdaptableScheduler extends ProbeSchedulerBase {
  private val schedulerWeightRandomGenerator = Random

  //各策略权重的最大浮动，一方面是某个策略实际收益的评估，另一方面是控制系统扰动的因子
  private val maxWeightFluctuationRatio = 0.1
  //各策略权重的历史遗忘因子
  private val weightForgottenFactor = 0.2

  //策略集合
  val schedulerArray: Array[ProbeSchedulerBase] = Array[ProbeSchedulerBase](new CrossoverScheduler, new MutationScheduler, new RegressionScheduler,
    new BoundaryExpandScheduler, new DesertPlowScheduler, new RegressionIgniteScheduler, new MutationIgniteScheduler)
  //各策略初始权重
  private val schedulerWeights = Array.fill(schedulerArray.length)(1.0)
  //参数与策略的cache，用于对每类策略的实际收益进行跟踪
  private val paramSchedulerCache = new ConcurrentHashMap[IndexedSeq[Double], Int]

  /**
    * 更新各策略的最大的评估容忍限度比率
    *
    * @param maxEstimateAcceptRatio 新的各策略最大的评估容忍限度比率
    */
  override def setMaxEstimateAcceptRatio(maxEstimateAcceptRatio: Double) {
    super.setMaxEstimateAcceptRatio(maxEstimateAcceptRatio)
    schedulerArray.foreach(_.setMaxEstimateAcceptRatio(maxEstimateAcceptRatio))
  }

  /**
    * 增大系统扰动
    */
  override def amplifyFluctuation() {
    schedulerArray.foreach(_.amplifyFluctuation())
  }

  override def setLearner(learner: LearnerBase) {
    this.learner = learner
    schedulerArray.foreach(_.setLearner(learner))
  }

  /**
    * 更新参数对应策略的实际收益，并将该参数对应cache里的条目移除
    *
    * @param params     参数集合
    * @param validation 该参数集合的实际收益
    */
  private def updateParamSchedulerCache(params: IndexedSeq[Double], validation: Double) {
    if (paramSchedulerCache.containsKey(params)) {
      val schedulerIndex = paramSchedulerCache.remove(params)
      schedulerWeights.synchronized {
        schedulerWeights(schedulerIndex) = (1 - weightForgottenFactor) * schedulerWeights(schedulerIndex) + weightForgottenFactor * validation
      }
    }
  }

  /**
    * 记录参数与某个策略的对应关系，以便下次更新该策略的实际效益
    *
    * @param params         参数集合
    * @param schedulerIndex 产生该参数的策略索引
    */
  private def insertIntoParamSchedulerCache(params: IndexedSeq[Double], schedulerIndex: Int) {
    paramSchedulerCache.put(params, schedulerIndex)
  }

  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(currentTask: ProbeTask): Array[Double] = null

  /**
    * 获取下次要probe的任务，主要是获取新的要probe的超参数，该方法为框架性方法，是提供给外部获取任务的接口，子类无需重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的任务
    */
  override def getNextProbeTask(currentTask: ProbeTask): ProbeTask = {
    //任务运行点置0
    currentTask.runPoint = 0

    val currentParams = currentTask.getParams
    val currentEstimate = currentTask.getFinalValidation

    updateParamSchedulerCache(currentParams.toIndexedSeq, currentEstimate)

    //如果只有一个超参数，就禁用掉crossover策略
    if (currentParams.length <= 1) schedulerWeights(0) = 0.0

    var nextParams = currentParams
    var nextEstimate = 0.0
    var schedulerIndex = -1
    var maxEstimateAcceptRatioNice = maxEstimateAcceptRatio
    var sinkingTimes = 0
    do {
      //按照各策略历史收益进行抽样选择
      var minWeight = schedulerWeights.min
      minWeight -= 1E-6 * minWeight
      val weights = schedulerWeights.map {
        wIt =>
          val weight = SampleUtil.getNextGaussian(schedulerWeightRandomGenerator, wIt - minWeight, maxWeightFluctuationRatio)
          if (weight < 0.0) 0.0 else weight
      }
      schedulerIndex = SampleUtil.rouletteLikeSelect(schedulerWeightRandomGenerator, weights)
      val scheduler = schedulerArray(schedulerIndex)

      nextParams = scheduler.getNextParams(currentTask)
      if (null == nextParams) {
        //如果参数返回为空，说明该策略已失效，就禁用掉该策略
        schedulerWeights(schedulerIndex) = 0.0
        nextEstimate = 0.0
      } else if (!ParamHoldler.isUniqueParam(currentParams, nextParams)) {
        //判断新参数和历史参数是否相似，如果太相似，就重新获取新参数，同时增大系统扰动
        amplifyFluctuation()
        nextEstimate = 0.0
      } else {
        nextEstimate = learner.predict(nextParams)
        /*
        如果评估值大于当前任务的验证值，或者虽然小于但仍然在容许的限度范围内，就认为新生成的参数集合是可探测的,
        并且如果策略生成的参数评估多次失败，就线性提升容忍度（临时）
         */
        maxEstimateAcceptRatioNice = maxEstimateAcceptRatio + (1 - maxEstimateAcceptRatio) * sinkingTimes / maxSinkingTimes
        sinkingTimes += 1
      }
    } while (nextEstimate < currentEstimate && 1 - nextEstimate / currentEstimate > maxEstimateAcceptRatioNice * randomGenerator.nextDouble)

    currentTask.updateParams(nextParams)
    insertIntoParamSchedulerCache(currentTask.getParams.toIndexedSeq, schedulerIndex)
    currentTask
  }
}
