package org.automl.model.strategy.scheduler

import org.automl.model.context.ParamHoldler
import org.automl.model.operators.BaseOperator
import org.automl.model.strategy.ProbeTask

/**
  * Created by zhangyikuo on 2016/8/23.
  */
class RegressionScheduler extends ProbeSchedulerBase {
  //第一次探测结束后探测的步幅比率
  private val initScorePaceRatio = 0.1

  //paceAmplifyFactor退火率，默认是10次退火到一半
  private val paceAmplificationAnnealRatio = math.pow(0.5, 0.1)

  //每次探测步幅的放大因子，扩大步幅，增大扰动
  private var paceAmplifyFactor = 1.0

  /**
    * 增大系统扰动
    */
  override def amplifyFluctuation() {
    super.amplifyFluctuation()
    paceAmplifyFactor *= 2
  }

  /**
    * 减小系统扰动
    */
  private def easeFluctuation() {
    paceAmplifyFactor = math.max(1.0, paceAmplifyFactor * paceAmplificationAnnealRatio)
  }

  /**
    * 获取下次要probe的超参数列表，子类需要重写该方法
    *
    * @param currentTask 当前probe任务
    * @return 下次要probe的超参数列表
    */
  override def getNextParams(currentTask: ProbeTask): Array[Double] = {
    //轮盘选择要进行繁衍的某条线
    val paramMatrix = ParamHoldler.getBestParams
    val chosenRow = paramMatrix(choosePropagationLine(paramMatrix))
    val chosenRowValue: Double = chosenRow.last

    //获取与当前搜索线验证值差异值最大的和最小的差异值
    val diffTuple = paramMatrix.foldLeft((Double.MaxValue, Double.MinValue)) {
      case ((min, max), paramArray) =>
        val diff = math.abs(paramArray.last - chosenRowValue)
        if (diff <= 0) (min, max) else if (diff < min) (diff, max) else if (diff > max) (min, diff) else (min, max)
    }

    //按照距离当前搜索线的距离（验证值度量），对其周围的数据进行分化学习，学习出当前线周围的local function
    val localLearner = learner.clone
    var nextPace = 0.0
    if (Double.MaxValue != diffTuple._1 && Double.MinValue != diffTuple._2 && diffTuple._1 != diffTuple._2) {
      var weightSum = 0.0
      paramMatrix.foreach {
        paramArray =>
          val diff = math.abs(paramArray.last - chosenRowValue)
          //按照距离当前搜索线的距离（验证值度量），对其周围的数据计算权重
          val weight = (diffTuple._2 - diff) / diffTuple._2
          weightSum += weight
          //对当前线周围搜索线进行加权模糊，求出适当的下步搜索步幅，该搜索步幅衡量chosenRow周围验证值变化的大小
          nextPace += weight * diff
          if (weight > randomGenerator.nextDouble) localLearner.onlineLearn(paramArray)
      }
      nextPace /= weightSum
    } else nextPace = chosenRowValue * initScorePaceRatio
    nextPace *= paceAmplifyFactor
    easeFluctuation()

    //对计算得到的下步搜索步幅（验证值），按照lcoalLearner的权重进行参数步幅分配，分配的方式按照欧式空间中的欧式长度进行分解
    val paramWeights = localLearner.getWeights
    //计算欧式长度平方
    val paramWeightSum = paramWeights.foldLeft(0.0) {
      case (norm, weight) => norm + weight * weight
    }
    var paramIndex = -1
    val operatorChain = currentTask.getOperatorChain
    operatorChain.flatMap {
      operator =>
        for (j <- 0 until operator.getParamNum) yield {
          paramIndex += 1
          val (bottom, upper) = operator.getParamBoundary(null, j)
          //按照欧式空间中的欧式长度（nextPace）进行分解
          var param = chosenRow(paramIndex) + paramWeights(paramIndex) * nextPace / paramWeightSum
          param = if (param > upper) upper else if (param < bottom) bottom else param
          if (BaseOperator.PARAM_TYPE_BOOLEAN == operator.getParamType(j) || BaseOperator.PARAM_TYPE_INT == operator.getParamType(j))
            math.round(param)
          else param
        }
    }
  }
}
