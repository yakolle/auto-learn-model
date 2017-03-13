package org.automl.model.context

import org.automl.model.operators.BaseOperator
import org.automl.model.strategy.ProbeTask
import org.automl.model.utils.MathUtil

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangyikuo on 2017/3/13.
  */
object ParamHandler {
  //超参数matrix，目前为止搜索的所有的超参数集合，最后一列是实际验证值
  private val params: ArrayBuffer[Array[Double]] = new ArrayBuffer[Array[Double]]
  //超参数matrix各行到原点的距离
  private val paramDistances: ArrayBuffer[Double] = new ArrayBuffer[Double]

  //各超参数的范围
  private var paramBoundaries: Array[(Double, Double)] = _

  //目前为止效果最好的搜索任务，第一个元素是算子序列（用算子序列，是方便取参数包括算子里别的属性），第二个是该次探索的实际验证值
  private var bestOperatorSequences: Array[(Array[BaseOperator], Double)] = _

  def getParams = params.toArray

  def getParamBoundaries = this.paramBoundaries

  def setParamBoundaries(paramBoundaries: Array[(Double, Double)]) = this.paramBoundaries = paramBoundaries

  /**
    * 更新超参数matrix
    *
    * @param param 本次超参数数组，最后一个是验证值
    */
  def updateParams(param: Array[Double]) {
    val norm = MathUtil.calcNorm(param.dropRight(1))

    params.synchronized {
      params += param

      paramDistances.synchronized {
        paramDistances += norm
      }
    }
  }

  /**
    * 获取agent一共探测了多少次
    *
    * @return 所有agent一共探测了多少次
    */
  def getRunTimes = params.size

  def getBestOperatorSequences = bestOperatorSequences

  def initBestOperatorSequences(size: Int) {
    bestOperatorSequences = new Array[(Array[BaseOperator], Double)](size)
    for (i <- 0 until size) bestOperatorSequences(i) = (null, Double.MinValue)
  }

  /**
    * 更新目前为止效果最好的搜索任务
    *
    * @param probeTask 本次探测结束后的任务对象
    */
  def updateBestOperatorSequences(probeTask: ProbeTask) {
    bestOperatorSequences.synchronized {
      //对最好集合里的验证值最小的一个进行更新，如果本次探测的效果更好的话
      val minTuple = bestOperatorSequences.foldLeft((Double.MaxValue, -1, 0)) {
        case ((min, minIndex, index), (operatorArray, value)) =>
          if (value < min) (value, index, index + 1) else (min, minIndex, index + 1)
      }
      val validation = probeTask.getFinalValidation
      if (validation > minTuple._1) bestOperatorSequences(minTuple._2) = (probeTask.getOperatorChain.map(_.clone), validation)
    }
  }

  /**
    * 每次探测任务结束后，对探测的结果进行反馈
    *
    * @param probeTask 本次探测结束后的任务对象
    */
  def feedback(probeTask: ProbeTask) {
    updateParams(probeTask.getParams :+ probeTask.getFinalValidation)
    updateBestOperatorSequences(probeTask)
  }

  /**
    * 获取目前为止搜索到的验证值最高的top beamSearchNum个超参数集合及相应的验证值
    *
    * @return 验证值最高的top beamSearchNum个超参数集合及相应的验证值
    */
  def getBestParams: Array[Array[Double]] = {
    for ((operatorChain, validation) <- bestOperatorSequences; if null != operatorChain) yield {
      val params = operatorChain.flatMap {
        operator =>
          for (i <- 0 until operator.getParamNum) yield operator.getCurrentParam(i)
      }
      params :+ validation
    }
  }
}
