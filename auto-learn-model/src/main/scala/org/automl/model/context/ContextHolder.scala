package org.automl.model.context

import java.io.{BufferedWriter, File, FileWriter, IOException}
import java.util

import org.apache.commons.io.FileUtils
import org.apache.spark.ml.attribute.{Attribute, AttributeGroup, NumericAttribute}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.automl.model.operators.BaseOperator
import org.automl.model.operators.data.evaluation.EvaluationBase
import org.automl.model.operators.data.sift.SiftFeaturesBase
import org.automl.model.operators.data.transform.TransformBase
import org.automl.model.operators.model.train.TrainBase
import org.automl.model.operators.model.validation.ValidationBase
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
  //各超参数的范围
  private var paramBoundaries: Array[(Double, Double)] = _
  //最理想情况下的验证值
  private var idealValidation: Double = 0.0

  private var lastSteadyTimes = 0
  private var lastSteadyClusterMeanDist = Double.MaxValue
  private var currentSteadyTimes = 0
  private var lastClusterMeanDist = Double.MaxValue

  //收敛记录
  private val convergeRecBuffer: ArrayBuffer[(Int, Double, Array[Array[Double]])] = new ArrayBuffer[(Int, Double, Array[Array[Double]])]

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
    for ((operatorChain, validation) <- bestOperatorSequences; if null != operatorChain) yield {
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
    * 输出收敛记录
    *
    * @param outputFilePath 收敛记录文件路径
    */
  def outputConvergenceRecord(outputFilePath: String) {
    val lines = new java.util.ArrayList[String]
    val strBuffer = StringBuilder.newBuilder
    for (learnRec <- convergeRecBuffer) {
      strBuffer.clear()
      strBuffer.append(learnRec._1).append("\t").append(learnRec._2).append("\t")
      for (bestParams <- learnRec._3) {
        for (bestParamEle <- bestParams) strBuffer.append(bestParamEle).append(",")

        strBuffer.setLength(strBuffer.length - 1)
        strBuffer.append("\t")
      }

      lines.add(strBuffer.substring(0, strBuffer.length - 1))
    }

    try
      FileUtils.writeLines(new File(outputFilePath), lines)
    catch {
      case e: IOException =>
        e.printStackTrace()
    }
  }

  /**
    * 输出搜索到的最好结果
    *
    * @param outputFilePath 搜索结果文件路径
    */
  def outputBestSearchResults(outputFilePath: String) {
    val writer = new BufferedWriter(new FileWriter(outputFilePath))
    val strBuffer = StringBuilder.newBuilder

    try {
      bestOperatorSequences.foreach {
        case (operatorChain, validation) =>
          if (null != operatorChain) {
            operatorChain.foreach {
              case operator: EvaluationBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()

                strBuffer.clear()
                operator.getEvaluations.foreach(strBuffer.append(_).append("\t"))
                writer.write(strBuffer.substring(0, strBuffer.length - 1))
                writer.newLine()

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case operator: TransformBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()
                writer.write(if (operator.isOn) "on" else "off")
                writer.newLine()

                if (operator.isOn) {
                  operator.explain(writer)
                  writer.newLine()
                }

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case operator: SiftFeaturesBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()

                strBuffer.clear()
                operator.getFeatureIDs.foreach(strBuffer.append(_).append("\t"))
                writer.write(strBuffer.substring(0, strBuffer.length - 1))
                writer.newLine()

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case operator: TrainBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()

                operator.explainModel(writer)
                writer.newLine()

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case operator: ValidationBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()

                strBuffer.clear()
                operator.getValidations.foreach {
                  case (trainValidation, testValidation) =>
                    strBuffer.append(trainValidation).append(",").append(testValidation).append("\t")
                }
                writer.write(strBuffer.substring(0, strBuffer.length - 1))
                writer.newLine()

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case _ =>
            }

            writer.write("finalValidation=" + validation)
            writer.newLine()
          }
          writer.write("===========================================================")
          writer.newLine()
          writer.newLine()
      }

      writer.close()
    } catch {
      case e: IOException =>
        e.printStackTrace()
    }
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

    //验证值的权重比例随着运行时间的增加逐渐增加，以防止非凸且极大值都差不多的情景下，参数变化剧烈导致无法收敛的问题
    val validationWeight = TaskBuilder.initValidationWeight +
      (TaskBuilder.maxValidationWeight - TaskBuilder.initValidationWeight) / (1.0 + math.exp(-getRunTimes / 64.0 + 3.7))

    //x^2=a,t^2=1-a，其中a=TaskBuilder.validationWeight，x=validationWeight，t为原超参数变换后的norm
    weights = if (0 == weightNorm) Array.fill(weights.length)(math.sqrt((1 - validationWeight) / weights.length))
    else {
      val weightFactor = math.sqrt(1 - validationWeight) / weightNorm
      weights.map(_ * weightFactor) :+ math.sqrt(validationWeight)
    }

    //归一化
    val bestParams = getBestParams.map {
      params =>
        (for (i <- paramBoundaries.indices) yield {
          val bottom = paramBoundaries(i)._1
          val upper = paramBoundaries(i)._2
          (params(i) - bottom) / (upper - bottom)
        }).toArray[Double] :+ (params.last / this.idealValidation)
    }
    //计算这些超参数数据（将这些超参数看成点）的中心
    val bestParamsCenter = MathUtil.calcMean(bestParams)
    //计算各条线到中心的平均距离，并且按照当前学习器学习到的各超参数的评估权重进行加权
    val clusterMeanDist = bestParams.map(SimilarityUtil.calcWeightedEuclideanDistance(_, bestParamsCenter, weights)).sum / bestParams.length

    convergeRecBuffer += ((this.getRunTimes, clusterMeanDist, bestParams))

    //如果各条线到中心的平均距离小于某个阈值，并且稳定（变化不大）次数达到一定阈值，就认为是收敛了
    val converged = if (clusterMeanDist <= TaskBuilder.convergedThreshold) {
      val lastSteadyDistDiff = math.abs(clusterMeanDist - lastSteadyClusterMeanDist)
      if (lastSteadyDistDiff <= TaskBuilder.convergedTolerance) {
        lastSteadyTimes += 1
        lastSteadyClusterMeanDist = clusterMeanDist
      } else lastSteadyTimes = (lastSteadyTimes * TaskBuilder.steadyTimeDiveRatio).toInt
      if (0 == lastSteadyTimes) lastSteadyClusterMeanDist = clusterMeanDist

      val curDistDiff = math.abs(clusterMeanDist - lastClusterMeanDist)
      if (curDistDiff <= TaskBuilder.convergedTolerance) currentSteadyTimes += 1
      else {
        if (currentSteadyTimes >= lastSteadyTimes) {
          lastSteadyClusterMeanDist = lastClusterMeanDist
          lastSteadyTimes = currentSteadyTimes
        }
        currentSteadyTimes = 0
      }

      math.max(lastSteadyTimes, currentSteadyTimes) > TaskBuilder.maxSteadyTimes
    } else false
    lastClusterMeanDist = clusterMeanDist

    converged
  }
}
