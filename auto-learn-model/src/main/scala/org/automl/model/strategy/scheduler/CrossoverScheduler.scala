package org.automl.model.strategy.scheduler

import org.automl.model.strategy.ProbeTask

import scala.util.Random

/**
  * Created by zhangyikuo on 2016/8/23.
  */
class CrossoverScheduler extends ProbeSchedulerBase {
  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param randomGenerator 随机源生成器
    * @param currentTask     当前probe任务
    * @param paramMatrix     超参数数据
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(randomGenerator: Random, currentTask: ProbeTask, paramMatrix: Array[Array[Double]]): Array[Double] = {
    val currentParams = currentTask.getParams
    val paramLen = currentParams.length
    if (paramLen <= 1) currentParams
    else {
      //随机选择交叉点
      var crossoverPoint = randomGenerator.nextInt(paramLen)
      //根据参数序列中的交叉点找到该参数所属算子的算子索引
      currentTask.runPoint = currentTask.getRunPoint(crossoverPoint)
      if (0 == crossoverPoint) crossoverPoint = 1

      //轮盘选择与历史搜索线中的那条线进行交叉，轮盘选择的依据是历史搜索线的时间验证值
      val peerIndex = choosePropagationLine(randomGenerator, paramMatrix)
      currentParams.take(crossoverPoint) ++ paramMatrix(peerIndex).slice(crossoverPoint, paramLen)
    }
  }
}
