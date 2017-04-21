package org.automl.model.context

/**
  * Created by zhangyikuo on 2017/4/21.
  */
class ConvergeEstimator {
  // 目标函数收敛判断阈值
  var tolerance: Double = 1E-5
  // 稳态次数阈值，如果系统稳定判断的次数大于该阈值就认为是收敛了
  var steadyTimeThreshold = 10
  // 系统震荡次数阈值，如果系统波动的次数大于该阈值就认为系统正处于震荡状态
  var fluctuationTimeThreshold = 5
  // 系统波动次数衰减系数，如果系统暂时处于稳定状态，对过去波动次数的ease系数
  var fluctuationTimeDiveRatio = 0.5

  private var lastTarget = Double.MaxValue
  private var curSteadyTimes = 0
  private var lastDirection = 1
  private var curFluctuationTimes = 0

  /**
    * 当前系统状态判断
    *
    * @param curTarget 当前目标值
    * @return (是否收敛，是否处于波动状态)
    */
  def converged(curTarget: Double): (Boolean, Boolean) = {
    val diff = curTarget - lastTarget
    curSteadyTimes = if (math.abs(diff) <= tolerance) curSteadyTimes + 1 else 0
    lastTarget = curTarget

    val curDirection = if (diff > 0) 1 else -1
    curFluctuationTimes = if (curDirection != lastDirection) curFluctuationTimes + 1
    else (curFluctuationTimes * fluctuationTimeDiveRatio).toInt
    lastDirection = curDirection

    (curSteadyTimes > steadyTimeThreshold, curFluctuationTimes > fluctuationTimeThreshold)
  }
}
