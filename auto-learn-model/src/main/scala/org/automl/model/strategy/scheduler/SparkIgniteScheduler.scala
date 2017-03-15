package org.automl.model.strategy.scheduler

import org.automl.model.context.ParamHoldler
import org.automl.model.strategy.ProbeTask

/**
  * Created by zhangyikuo on 2017/3/15.
  */
class SparkIgniteScheduler extends ProbeSchedulerBase {
  /**
    * 获取距离最远的两个参数
    *
    * @return 距离最远的两个参数
    */
  protected def getFarthestParams = ParamHoldler.getFarthestParams

  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(currentTask: ProbeTask): Array[Double] = {
    val (param1, param2) = getFarthestParams
    (for (i <- 0 until param1.length - 1) yield (param1(i) + param2(i)) / 2.0).toArray
  }
}
