package org.automl.model.context

import java.util

import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.automl.model.output.OutputHandler
import org.automl.model.strategy.learn.LearnerBase
import org.automl.model.strategy.scheduler.ProbeSchedulerBase

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangyikuo on 2016/9/1.
  */
object ContextHolder {
  private var sparkSession: SparkSession = _
  private var learner: LearnerBase = _
  private var scheduler: ProbeSchedulerBase = _

  private var lastMaxValidation = 0.0
  private var steadyTimes = 0

  //收敛记录，格式为(runTimes, steadyTimes, maxEstimateAcceptRatio, bestParams)
  private val convergeRecBuffer: ArrayBuffer[(Int, Int, Double, Array[Array[Double]])] = new ArrayBuffer[(Int, Int, Double,
    Array[Array[Double]])](TaskBuilder.convergeRecBufferSize)

  def getConvergeRecords = convergeRecBuffer

  def setSparkSession(sparkSession: SparkSession) {
    if (null == this.sparkSession) this.sparkSession = sparkSession
  }

  def setLearner(learner: LearnerBase) {
    if (null == this.learner) this.learner = learner
  }

  def setScheduler(scheduler: ProbeSchedulerBase) {
    if (null == this.scheduler) this.scheduler = scheduler
  }

  /**
    * 创建数据schema
    *
    * @param featuresLen 特征数
    * @return schema（只有两列，一列为features——向量形式，一列为label列）
    */
  def buildSchema(featuresLen: Int): StructType = {
    val featuresAttrs = Array.fill(featuresLen)(NumericAttribute.defaultAttr)
    StructType(Array(new AttributeGroup("features", featuresAttrs.asInstanceOf[Array[Attribute]]).toStructField,
      StructField("label", DoubleType, nullable = false)))
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

    sparkSession.createDataFrame(rowList, buildSchema(featuresLen))
  }

  /**
    * 按照当前搜索到的最好的搜索任务集合的状态，动态调整各个策略的评估容忍限度
    */
  def adjustMaxEstimateAcceptRatio() {
    var errSum = 0.0
    var validationSum = 0.0
    ParamHoldler.getBestOperatorSequences.foreach {
      case (operatorChain, validation) =>
        if (null != operatorChain) {
          val paramArray = operatorChain.flatMap(operator => for (i <- 0 until operator.getParamNum) yield operator.getCurrentParam(i))
          //计算评估残差
          errSum += math.abs(learner.predict(paramArray) - validation)
          validationSum += validation
        }
    }
    scheduler.setMaxEstimateAcceptRatio(errSum / validationSum)
  }

  /**
    * 判断是否收敛，根据当前搜索到的最好的搜索任务集合的状态进行判断
    *
    * @return 是否收敛
    */
  def hasConverged: Boolean = {
    var converged = false
    val curRunTimes = ParamHoldler.getRunTimes

    if (convergeRecBuffer.nonEmpty && curRunTimes <= convergeRecBuffer.last._1) Thread.sleep(TaskBuilder.learnInterval)
    else {
      val curMaxValidation = ParamHoldler.getBestOperatorSequences.maxBy(_._2)._2
      steadyTimes = if (curMaxValidation - lastMaxValidation <= TaskBuilder.validationTolerance) steadyTimes + 1 else 0

      convergeRecBuffer += ((curRunTimes, steadyTimes, scheduler.getMaxEstimateAcceptRatio, ParamHoldler.getBestParams))
      if (convergeRecBuffer.length >= TaskBuilder.convergeRecBufferSize) {
        OutputHandler.outputConvergenceRecord(convergeRecBuffer, TaskBuilder.getConvergenceRecordOutputPath)
        OutputHandler.outputBestSearchResults(ParamHoldler.getBestOperatorSequences, TaskBuilder.getBestResultsOutputPath)

        convergeRecBuffer.clear()
      }

      lastMaxValidation = curMaxValidation
      converged = steadyTimes > TaskBuilder.maxSteadyTimes
    }

    converged
  }
}