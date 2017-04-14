package org.automl.model.strategy.scheduler

import org.automl.model.context.ParamHoldler
import org.automl.model.operators.BaseOperator
import org.automl.model.strategy.ProbeTask
import org.automl.model.utils.SampleUtil

import scala.collection.mutable

/**
  * Created by zhangyikuo on 2017/3/11.
  */
class BoundaryExpandScheduler extends ProbeSchedulerBase {
  private val disableExpandParamIndices = mutable.Set.empty[Int]

  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(currentTask: ProbeTask): Array[Double] = {
    var weights = learner.getParamImportances
    if (disableExpandParamIndices.size >= weights.length) null
    else {
      val maxWeight = weights.max
      //减小学习器重要程度评估的影响，让每个超参数被扩展的机会相当（但不能相同）
      weights = (for (i <- weights.indices) yield if (disableExpandParamIndices.contains(i)) 0.0 else weights(i) + maxWeight).toArray

      val expandPoint = SampleUtil.rouletteLikeSelect(paramChooseRandomGenerator, weights)
      //根据参数序列中的边界扩展点找到该参数所属算子的算子索引
      currentTask.runPoint = currentTask.getRunPoint(expandPoint)
      val operator = currentTask.getOperatorChain.apply(currentTask.runPoint)
      //获取扩展点是该算子的第几个参数
      val offset = expandPoint - currentTask.getParamIndexRange(currentTask.runPoint)._1

      var param = 0.0
      val (curMinParam, curMaxParam) = ParamHoldler.getCurrentParamMinMax(expandPoint)
      if (BaseOperator.PARAM_TYPE_BOOLEAN == operator.getParamType(offset)) param = 1 - operator.getCurrentParam(offset)
      else {
        val fluctuation = operator.getEmpiricalParamPace(null, offset) * (1.0 + math.abs(randomGenerator.nextGaussian))
        param = if (randomGenerator.nextBoolean) curMaxParam + fluctuation else curMinParam - fluctuation
      }
      val (bottom, upper) = operator.getParamBoundary(null, offset)
      disableExpandParamIndices.synchronized {
        if (param >= upper && curMinParam == bottom) disableExpandParamIndices += expandPoint
        else if (param <= bottom && curMaxParam == upper) disableExpandParamIndices += expandPoint
      }

      val paramArray = currentTask.getParams
      paramArray(expandPoint) = param
      paramArray
    }
  }
}
