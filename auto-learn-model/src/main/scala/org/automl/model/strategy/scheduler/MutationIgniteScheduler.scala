package org.automl.model.strategy.scheduler

import org.automl.model.operators.BaseOperator
import org.automl.model.strategy.ProbeTask
import org.automl.model.utils.SampleUtil

/**
  * Created by zhangyikuo on 2017/3/15.
  */
class MutationIgniteScheduler extends SparkIgniteScheduler {
  //fluctuationAmplifyFactor退火率，默认是10次退火到一半
  private val fluctuationAnnealRatio = math.pow(0.5, 0.1)

  //mutation浮动的放大因子，增大扰动
  private var fluctuationAmplifyFactor = 1.0

  /**
    * 增大系统扰动
    */
  override def amplifyFluctuation() {
    super.amplifyFluctuation()
    fluctuationAmplifyFactor *= 2
  }

  /**
    * 减小系统扰动
    */
  private def easeFluctuation() {
    fluctuationAmplifyFactor = math.max(1.0, fluctuationAmplifyFactor * fluctuationAnnealRatio)
  }

  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(currentTask: ProbeTask): Array[Double] = {
    //获取sparkle点
    val param = super.getNextParams(currentTask)

    var weights = learner.getParamImportances
    val maxWeight = weights.max
    //减小学习器重要程度评估的影响，让每个超参数被扩展的机会相当（但不能相同）
    weights = weights.map(_ + maxWeight)

    val mutationPoint = SampleUtil.rouletteLikeSelect(paramChooseRandomGenerator, weights)
    //根据参数序列中的变异点找到该参数所属算子的算子索引
    val runPoint = currentTask.getRunPoint(mutationPoint)
    val operator = currentTask.getOperatorChain.apply(runPoint)

    //获取变异点是该算子的第几个参数
    val offset = mutationPoint - currentTask.getParamIndexRange(runPoint)._1
    val paramEle = if (BaseOperator.PARAM_TYPE_BOOLEAN == operator.getParamType(offset)) 1 - operator.getCurrentParam(offset)
    else {
      //计算变异幅度
      val fluctuation = operator.getEmpiricalParamPace(null, offset) * (fluctuationAmplifyFactor +
        math.abs(randomGenerator.nextGaussian))
      easeFluctuation()
      (if (randomGenerator.nextBoolean) fluctuation else -fluctuation) + param(mutationPoint)
    }

    param(mutationPoint) = paramEle
    param
  }
}
