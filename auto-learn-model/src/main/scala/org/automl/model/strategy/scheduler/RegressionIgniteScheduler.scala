package org.automl.model.strategy.scheduler

import org.automl.model.strategy.ProbeTask

/**
  * Created by zhangyikuo on 2017/3/15.
  */
class RegressionIgniteScheduler extends SparkIgniteScheduler {
  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(currentTask: ProbeTask): Array[Double] = {
    val (param1, param2) = getFarthestParams
    val param = (for (i <- 0 until param1.length - 1) yield (param1(i) + param2(i)) / 2.0).toArray
    val nextPace = math.abs(param1.last - param2.last) / 2.0

    //对nextPace，按照learner的权重进行参数步幅分配，分配的方式按照欧式空间中的欧式长度进行分解
    val paramWeights = learner.getWeights
    //计算欧式长度平方
    val paramWeightSum = paramWeights.foldLeft(0.0) {
      case (norm, weight) => norm + weight * weight
    }
    var paramIndex = -1
    currentTask.getOperatorChain.flatMap {
      operator =>
        for (j <- 0 until operator.getParamNum) yield {
          paramIndex += 1
          //按照欧式空间中的欧式长度（nextPace）进行分解
          param(paramIndex) + paramWeights(paramIndex) * nextPace / paramWeightSum
        }
    }
  }
}
