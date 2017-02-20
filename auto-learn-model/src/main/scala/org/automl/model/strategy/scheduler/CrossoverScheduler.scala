package org.automl.model.strategy.scheduler

import org.automl.model.strategy.ProbeTask
import org.automl.model.utils.SampleUtil

import scala.util.Random

/**
  * Created by zhangyikuo on 2016/8/23.
  */
class CrossoverScheduler extends ProbeSchedulerBase {
  //学习器中各参数权重的最大浮动，一方面是学习器的可信度配置（可动态调整），另一方面是加大系统扰动的可控因子
  private val maxCrossoverPointFluctuationRatio = 0.1

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
    if (currentParams.length <= 1) currentParams
    else {
      //对交叉点进行抽样，将学习器的各参数权重看成各参数重要程度的概率
      val weights = learner.getWeights.map(wIt => SampleUtil.getNextNonNegativeTrimmedGaussian(randomGenerator, math.abs(wIt),
        maxCrossoverPointFluctuationRatio / 3))
      var crossoverPoint = SampleUtil.rouletteLikeSelect(weights)
      //根据参数序列中的交叉点找到该参数所属算子的算子索引
      currentTask.runPoint = currentTask.getRunPoint(crossoverPoint)
      //重新调整交叉点，尽量在重要的参数后面进行交叉
      val crossoverPointRange = currentTask.getParamIndexRange(currentTask.runPoint)
      crossoverPoint = crossoverPointRange._2
      //如果交叉点位于参数最末尾，就在参数前面进行交叉
      if (crossoverPoint >= weights.length - 1) {
        crossoverPoint = if (crossoverPointRange._1 == crossoverPointRange._2) {
          currentTask.runPoint = currentTask.getRunPoint(crossoverPoint - 1)
          currentTask.getParamIndexRange(currentTask.runPoint)._2
        } else crossoverPointRange._1
      }

      //轮盘选择与历史搜索线中的那条线进行交叉，轮盘选择的依据是历史搜索线的时间验证值
      val peerIndex = choosePropagationLine(randomGenerator, paramMatrix)
      currentParams.take(crossoverPoint + 1) ++ paramMatrix(peerIndex).slice(crossoverPoint + 1, weights.length)
    }
  }
}
