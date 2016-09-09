package org.automl.model.strategy.scheduler

import org.automl.model.operators.BaseOperator
import org.automl.model.strategy.ProbeTask
import org.automl.model.utils.SampleUtil

import scala.util.Random

/**
  * Created by zhangyikuo on 2016/8/23.
  */
class MutationScheduler extends ProbeSchedulerBase {
  //学习器中各参数权重的最大浮动，一方面是学习器的可信度配置（可动态调整），另一方面是加大系统扰动的可控因子
  private val maxMutationPointFluctuationRatio = 0.1

  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param randomGenerator 随机源生成器
    * @param currentTask     当前probe任务
    * @param paramMatrix     超参数数据
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(randomGenerator: Random, currentTask: ProbeTask, paramMatrix: Array[Array[Double]]): Array[Double] = {
    //对变异点进行抽样，将学习器的各参数权重看成各参数重要程度的概率
    val weights = learner.getWeights.map(wIt => SampleUtil.getNextNonNegativeTrimmedGaussian(randomGenerator, wIt, maxMutationPointFluctuationRatio / 3))
    val mutationPoint = SampleUtil.rouletteLikeSelect(weights)
    //根据参数序列中的变异点找到该参数所属算子的算子索引
    currentTask.runPoint = currentTask.getRunPoint(mutationPoint)
    val operator = currentTask.getOperatorChain.apply(currentTask.runPoint)

    //获取变异点是该算子的第几个参数
    val offset = mutationPoint - currentTask.getParamIndexRange(currentTask.runPoint)._1
    var param = 0.0
    if (BaseOperator.PARAM_TYPE_BOOLEAN == operator.getParamType(offset)) param = 1 - operator.getCurrentParam(offset)
    else {
      //计算变异幅度，实际上是对当前参数值和该参数经验值的一种动态抽样，从该策略对所有的参数调整的角度考虑，实际上是一种Gibbs的变异抽样
      var fluctuation = math.abs(operator.getCurrentParam(offset) - operator.getEmpiricalParam(null, offset)) +
        operator.getEmpiricalParamPace(null, offset) * math.abs(randomGenerator.nextGaussian)
      fluctuation = if (randomGenerator.nextBoolean) randomGenerator.nextDouble * fluctuation else -randomGenerator.nextDouble * fluctuation
      param = operator.getEmpiricalParam(null, offset) + fluctuation
      val (bottom, upper) = operator.getParamBoundary(null, offset)
      param = if (param > upper) upper else if (param < bottom) bottom else param
      if (BaseOperator.PARAM_TYPE_INT == operator.getParamType(offset)) param = math.round(param)
    }

    val paramArray = currentTask.getParams
    paramArray(mutationPoint) = param
    paramArray
  }
}
