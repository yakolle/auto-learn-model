package org.automl.model.strategy.scheduler

import org.automl.model.context.ParamHoldler
import org.automl.model.strategy.ProbeTask
import org.automl.model.utils.{SampleUtil, SparsityUtil}

import scala.collection.mutable

/**
  * Created by zhangyikuo on 2017/3/11.
  */
class DesertPlowScheduler extends ProbeSchedulerBase {
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

      val plowPoint = SampleUtil.rouletteLikeSelect(paramChooseRandomGenerator, weights)
      //根据参数序列中的边界扩展点找到该参数所属算子的算子索引
      currentTask.runPoint = currentTask.getRunPoint(plowPoint)
      val operator = currentTask.getOperatorChain.apply(currentTask.runPoint)
      //获取扩展点是该算子的第几个参数
      val offset = plowPoint - currentTask.getParamIndexRange(currentTask.runPoint)._1

      //找到plowPoint对应参数列间隔最大的两点
      val plowParam = ParamHoldler.getParams.map(_ (plowPoint))
      val (curMinParam, curMaxParam) = (plowParam.min, plowParam.max)
      val (isBalanced, paramIndex1, paramIndex2) = SparsityUtil.findMaxGap(plowParam, curMinParam, curMaxParam)
      val param = (plowParam(paramIndex1) + plowParam(paramIndex2)) / 2.0

      val (bottom, upper) = operator.getParamBoundary(null, offset)
      if (isBalanced && curMinParam == bottom && curMaxParam == upper) disableExpandParamIndices.synchronized {
        disableExpandParamIndices += plowPoint
      }

      val paramArray = currentTask.getParams
      paramArray(plowPoint) = param
      paramArray
    }
  }
}
