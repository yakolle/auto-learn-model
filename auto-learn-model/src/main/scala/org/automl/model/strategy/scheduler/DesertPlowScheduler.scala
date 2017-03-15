package org.automl.model.strategy.scheduler

import org.automl.model.context.ParamHoldler
import org.automl.model.operators.BaseOperator
import org.automl.model.strategy.ProbeTask
import org.automl.model.utils.SampleUtil

import scala.collection.mutable

/**
  * Created by zhangyikuo on 2017/3/11.
  */
class DesertPlowScheduler extends ProbeSchedulerBase {
  private val disableExpandParamIndices = mutable.Set.empty[Int]

  /**
    * 找到间隔最大的两个参数的平均值
    *
    * @param plowParam   参数列表
    * @param curMinParam 参数最小值
    * @param curMaxParam 参数最大值
    * @return 返回值格式为:(是否均衡,间隔最大的两个参数的平均值)
    */
  private def findMaxGap(plowParam: Array[Double], curMinParam: Double, curMaxParam: Double) = {
    //构建plowParam.length个桶，将plowParam装入相应桶中
    val buckets: Array[(Double, Double)] = Array.fill(plowParam.length)(null)
    plowParam.foreach {
      paramEle =>
        val sectionIndex = math.min(((paramEle - curMinParam) * plowParam.length / (curMaxParam - curMinParam)).toInt, plowParam.length - 1)
        val sectionTuple = buckets(sectionIndex)
        buckets(sectionIndex) = if (null == sectionTuple) (paramEle, paramEle)
        else if (paramEle > sectionTuple._2) (sectionTuple._1, paramEle)
        else if (paramEle < sectionTuple._1) (paramEle, sectionTuple._2)
        else sectionTuple
    }

    //找到连续空桶数最多的section
    var maxEmptyBucketNum = 0
    var paramStart = 0
    var paramEnd = 0
    var curEmptyBucketNum = 0
    var curEmptyBucketStart = 0
    for (i <- buckets.indices) {
      if (null == buckets(i)) curEmptyBucketNum += 1
      else {
        if (curEmptyBucketNum >= maxEmptyBucketNum) {
          maxEmptyBucketNum = curEmptyBucketNum
          paramStart = curEmptyBucketStart
          paramEnd = i
        }
        curEmptyBucketNum = 0
        curEmptyBucketStart = i
      }
    }

    //如果没有空桶，实际上buckets已是排好序的plowParam了
    if (paramEnd - paramStart <= 1) {
      var maxSpan = Double.MinValue
      for (i <- 1 until buckets.length) {
        val curSpan = buckets(i)._1 - buckets(i - 1)._1
        if (curSpan > maxSpan) {
          paramEnd = i
          maxSpan = curSpan
        }
      }
      (true, buckets(paramEnd)._1 - maxSpan / 2.0)
    } else (false, (buckets(paramEnd)._1 - buckets(paramStart)._2) / 2.0)
  }

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
      var (isBalanced, param) = findMaxGap(plowParam, curMinParam, curMaxParam)

      if (BaseOperator.PARAM_TYPE_INT == operator.getParamType(offset) || BaseOperator.PARAM_TYPE_BOOLEAN == operator.getParamType(offset))
        param = math.round(param)
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
