package org.automl.model.strategy.scheduler

import org.automl.model.context.ParamHandler
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
    * @param paramMatrix 超参数数据
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(currentTask: ProbeTask, paramMatrix: Array[Array[Double]]): Array[Double] = {
    if (disableExpandParamIndices.size >= paramMatrix.head.length) null
    else {
      var weights = learner.getParamImportances
      weights = (for (i <- weights.indices) yield if (disableExpandParamIndices.contains(i)) 0.0 else math.abs(weights(i))).toArray
      val maxWeight = weights.max
      //减小学习器重要程度评估的影响，让每个超参数被扩展的机会相当（但不能相同）
      weights = weights.map(_ + maxWeight)

      val expandPoint = SampleUtil.rouletteLikeSelect(weights)
      //根据参数序列中的边界扩展点找到该参数所属算子的算子索引
      currentTask.runPoint = currentTask.getRunPoint(expandPoint)
      val operator = currentTask.getOperatorChain.apply(currentTask.runPoint)
      //获取扩展点是该算子的第几个参数
      val offset = expandPoint - currentTask.getParamIndexRange(currentTask.runPoint)._1

      var param = 0.0
      val (curMinParam, curMaxParam) = ParamHandler.getCurrentParamMinMax(expandPoint)
      if (BaseOperator.PARAM_TYPE_BOOLEAN == operator.getParamType(offset)) param = 1 - operator.getCurrentParam(offset)
      else {
        val fluctuation = operator.getEmpiricalParamPace(null, offset) * (1.0 + math.abs(randomGenerator.nextGaussian))
        param = if (randomGenerator.nextBoolean) curMaxParam + fluctuation else curMinParam - fluctuation
        if (BaseOperator.PARAM_TYPE_INT == operator.getParamType(offset)) param = math.round(param)
      }
      val (bottom, upper) = operator.getParamBoundary(null, offset)
      param = if (param >= upper) {
        if (curMinParam == bottom) disableExpandParamIndices += expandPoint
        upper
      } else if (param <= bottom) {
        if (curMaxParam == upper) disableExpandParamIndices += expandPoint
        bottom
      } else param

      val paramArray = currentTask.getParams
      paramArray(expandPoint) = param
      paramArray
    }
  }
}
