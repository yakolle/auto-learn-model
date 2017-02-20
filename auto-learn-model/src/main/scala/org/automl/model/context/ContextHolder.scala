package org.automl.model.context

import java.util

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.automl.model.operators.BaseOperator
import org.automl.model.strategy.ProbeTask
import org.automl.model.strategy.learn.LearnerBase
import org.automl.model.strategy.scheduler.ProbeSchedulerBase
import org.automl.model.utils.{MathUtil, SimilarityUtil}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangyikuo on 2016/9/1.
  */
object ContextHolder {
  private var sparkSession: SparkSession = _
  private var learner: LearnerBase = _
  private var scheduler: ProbeSchedulerBase = _

  //超参数matrix，目前为止搜索的所有的超参数集合，最后一列是实际验证值
  private var params: ArrayBuffer[Array[Double]] = new ArrayBuffer[Array[Double]]
  //目前为止效果最好的搜索任务，第一个元素是算子序列（用算子序列，是方便取参数包括算子里别的属性），第二个是该次探索的实际验证值
  private var bestOperatorSequences: Array[(Array[BaseOperator], Double)] = _

  private var paramBoundaries: Array[(Double, Double)] = _

  private var idealValidation: Double = 0.0

  def setIdealValidation(idealValidation: Double) {
    this.idealValidation = idealValidation
  }

  def setSparkSession(sparkSession: SparkSession) {
    if (null == this.sparkSession) this.sparkSession = sparkSession
  }

  def setLearner(learner: LearnerBase) {
    if (null == this.learner) this.learner = learner
  }

  def setScheduler(scheduler: ProbeSchedulerBase) {
    if (null == this.scheduler) this.scheduler = scheduler
  }

  def getParams = params.toArray

  def getParamBoundaries = this.paramBoundaries

  def setParamBoundaries(paramBoundaries: Array[(Double, Double)]) = this.paramBoundaries = paramBoundaries

  def initBestOperatorSequences(size: Int) {
    bestOperatorSequences = new Array[(Array[BaseOperator], Double)](size)
    for (i <- 0 until size) bestOperatorSequences(i) = (null, Double.MinValue)
  }

  /**
    * 更新超参数matrix
    *
    * @param param 本次超参数数组，最后一个是验证值
    */
  def updateParams(param: Array[Double]) {
    params.synchronized {
      params += param
    }
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
    * 将数据按照当前的sqlContext转换成DataFrame
    *
    * @param data 数据
    * @return 转换后的DataFrame
    */
  def toDF(data: Array[Array[Double]]): DataFrame = {
    val featuresLen = data.head.length - 1
    val rowList = new util.ArrayList[Row]
    data.foreach(rec => rowList.add(Row(Vectors.dense(rec.take(featuresLen)), rec.last)))


    val featuresAttrs = Array.fill(featuresLen)(NumericAttribute.defaultAttr)
    val schema = StructType(Array(new AttributeGroup("features", featuresAttrs.asInstanceOf[Array[Attribute]]).toStructField,
      StructField("label", DoubleType, nullable = false)))

    sparkSession.createDataFrame(rowList, schema)
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
    * 获取agent一共探测了多少次
    *
    * @return 所有agent一共探测了多少次
    */
  def getRunTimes = params.size

  /**
    * 获取目前为止搜索到的验证值最高的top beamSearchNum个超参数集合及相应的验证值
    *
    * @return 验证值最高的top beamSearchNum个超参数集合及相应的验证值
    */
  def getBestParams: Array[Array[Double]] = {
    bestOperatorSequences.map {
      case (operatorChain, validation) =>
        val params = operatorChain.flatMap {
          operator =>
            for (i <- 0 until operator.getParamNum) yield operator.getCurrentParam(i)
        }
        params :+ validation
    }
  }

  /**
    * 按照当前搜索到的最好的搜索任务集合的状态，动态调整各个策略的评估容忍限度
    */
  def adjustMaxEstimateAcceptRatio() {
    var errSum = 0.0
    var validationSum = 0.0
    bestOperatorSequences.foreach {
      case (operatorChain, validation) =>
        val paramArray = operatorChain.flatMap(operator => for (i <- 0 until operator.getParamNum) yield operator.getCurrentParam(i))
        //计算评估残差
        errSum += math.abs(learner.predict(paramArray) - validation)
        validationSum += validation
    }
    scheduler.setMaxEstimateAcceptRatio(errSum / validationSum)
  }

  /**
    * 判断是否收敛，根据当前搜索到的最好的搜索任务集合的状态进行判断
    *
    * @return 是否收敛
    */
  def hasConverged: Boolean = {
    //获取当前学习器对各个超参数的评估权重
    var weights = learner.getWeights.map(math.abs)
    //计算最终验证值的权重，按照TaskBuilder.validationWeight作为验证值同超参数集合的比重
    val weightNorm = MathUtil.calcNorm(weights)
    //x^2=a,t^2=1-a，其中a=TaskBuilder.validationWeight，x=validationWeight，t为原超参数变换后的norm
    weights = if (0 == weightNorm) Array.fill(weights.length)(math.sqrt((1 - TaskBuilder.validationWeight) / weights.length))
    else {
      val weightFactor = math.sqrt(1 - TaskBuilder.validationWeight) / weightNorm
      weights.map(_ * weightFactor) :+ math.sqrt(TaskBuilder.validationWeight)
    }

    val bestParams = getBestParams
    //计算这些超参数数据（将这些超参数看成点）的中心
    val bestParamsCenter = MathUtil.calcMean(bestParams)
    //计算各条线到中心的平均距离，并且按照当前学习器学习到的各超参数的评估权重进行加权
    val clusterMeanDist = bestParams.map(SimilarityUtil.calcWeightedEuclideanDistance(_, bestParamsCenter, weights)).sum / bestParams.length
    //计算各条线最大距离，并且按照当前学习器学习到的各超参数的评估权重进行加权
    val paramDomainScales = paramBoundaries.map(boundPair => boundPair._2 - boundPair._1)
    val maxDist = MathUtil.calcWeightedNorm(paramDomainScales :+ this.idealValidation, weights)
    val idealMaxDist = MathUtil.calcWeightedNorm(Array.fill(paramDomainScales.length)(paramDomainScales.max)
      :+ this.idealValidation, weights)

    //如果各条线到中心的平均距离与各条线最大距离的比重小于某个阈值，就认为是收敛了
    clusterMeanDist / maxDist <= TaskBuilder.convergedThreshold * maxDist / idealMaxDist
  }
}
