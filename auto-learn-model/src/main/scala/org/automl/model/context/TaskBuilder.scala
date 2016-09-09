package org.automl.model.context

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.automl.model.operators.BaseOperator
import org.automl.model.operators.data.bagging.ABBagging
import org.automl.model.operators.data.format.DataAssembler
import org.automl.model.operators.model.train.LogisticRegressionTrain
import org.automl.model.operators.model.validation.{AUCValidation, AssemblyValidation, ValidationBase}
import org.automl.model.strategy.learn.{LearnerBase, LinearRegressionLearner}
import org.automl.model.strategy.scheduler.{AdaptableScheduler, ProbeSchedulerBase}
import org.automl.model.strategy.{ProbeAgent, ProbeTask}

import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by zhangyikuo on 2016/9/6.
  */
object TaskBuilder {
  //学习器进行全部批量学习的时间间隔
  var learnInterval = 100 * 1000
  //最大搜索次数
  val maxIterations = 10000
  //最小搜索次数
  var minIterations = 100
  //参数集合对应的验证值在欧式空间中的权重比例（参数集合和该参数集合对应的验证值构成整个参数的欧式空间）
  var validationWeight = 0.5
  //是否收敛的评估占比阈值
  var convergedThreshold = 0.001

  def loadData(sparkSession: SparkSession, args: Array[String]): DataFrame = {
    sparkSession.read.option("header", value = true).option("inferSchema", value = true)
      .csv("E:\\work\\output\\ranking\\train_data.csv")
  }

  /**
    * 加载算子，算子可以用配置文件的形式加载
    *
    * @param args 算子配置的一些属性，或者算子配置文件的地址
    * @return 算子序列，agent会严格按照该算子序列的顺序执行算子
    */
  def loadOperators(args: Array[String]): Array[BaseOperator] = {
    val trainer = new LogisticRegressionTrain
    trainer.setValidators(Array[ValidationBase](new AUCValidation))
    Array[BaseOperator](new ABBagging, new DataAssembler, trainer, new AUCValidation)
  }

  /**
    * 初始化AssemblyValidation，主要是加载各验证项相关权重
    *
    * @param operators 算子列表，据此列表获取各验证算子验证权重
    */
  def initAssemblyValidation(operators: Array[BaseOperator]) {
    val validatorWeights = Array((2.0, 3.0, -2.0, 0.0, 0.0))
    AssemblyValidation.setValidatorWeights(validatorWeights)
    val trainValidatorWeights = Array((2.0, 3.0, -2.0, 0.0, 0.0))
    AssemblyValidation.setTrainValidatorWeights(trainValidatorWeights)
    val trainingAndTrainedValidationWeight = (1.0, 1.0, -1.0, 0.0, 0.0)
    AssemblyValidation.setTrainingAndTrainedValidationWeight(trainingAndTrainedValidationWeight)
  }

  /**
    * 创建初始的探测超参数任务
    *
    * @param operators 算子序列原型
    * @param data      数据
    * @param buildNum  并行探测的agent数量
    * @return 探测任务数组
    */
  def buildProbeTask(operators: Array[BaseOperator], data: DataFrame, buildNum: Int): Array[ProbeTask] = {
    val paramBoundaries = new ArrayBuffer[(Double, Double)]

    val randomGenerator = Random
    val tasks = for (i <- 1 to buildNum) yield {
      val newOperators = operators.map {
        operator =>
          val copy = operator.clone
          val params = for (j <- 0 until copy.getParamNum) yield {
            val boundaryPair = copy.getParamBoundary(null, j)
            //超参数边界对每条搜索线都是一样的，所以只需要计算一次
            if (1 == i) paramBoundaries += boundaryPair
            //对每条线的初始搜索点进行随机化（高斯随机）探索
            var param = copy.getEmpiricalParam(null, j) + copy.getEmpiricalParamPace(null, j) * randomGenerator.nextGaussian
            param = if (param > boundaryPair._2) boundaryPair._2 else if (param < boundaryPair._1) boundaryPair._1 else param
            if (BaseOperator.PARAM_TYPE_BOOLEAN == operator.getParamType(j) || BaseOperator.PARAM_TYPE_INT == operator.getParamType(j))
              math.round(param)
            else param
          }
          copy.updateParam(params.toArray)
          copy
      }
      new ProbeTask(newOperators, data)
    }
    //保存每个超参数的边界
    ContextHolder.setParamBoundaries(paramBoundaries.toArray)
    tasks.toArray
  }

  def buildProbeAgent(buildNum: Int): Array[ProbeAgent] = (for (i <- 1 to buildNum) yield new ProbeAgent).toArray

  /**
    * 获取并行搜索的任务数量，可更加当前计算资源进行动态计算
    *
    * @param sparkSession 计算环境
    * @return 并可行搜索的任务数量
    */
  def getBeamSearchNum(sparkSession: SparkSession) = 10

  /**
    * 创建参数学习评估器
    *
    * @return 新的参数学习评估器
    */
  def getLearner: LearnerBase = {
    val learner = new LinearRegressionLearner
    ContextHolder.setLearner(learner)
    learner
  }

  /**
    * 创建自适应策略，该策略可根据收益（最终验证值）动态调整不同策略的比重
    *
    * @return 策略对象
    */
  def getScheduler: ProbeSchedulerBase = {
    val scheduler = new AdaptableScheduler
    scheduler.learner = getLearner
    scheduler.schedulerArray.foreach(_.learner = scheduler.learner)
    ContextHolder.setScheduler(scheduler)
    scheduler
  }
}