package org.automl.model.strategy

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.operators.data.evaluation.EvaluationBase
import org.automl.model.operators.model.train.TrainBase
import org.automl.model.operators.model.validation.{AssemblyValidation, ValidationBase}

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangyikuo on 2016/8/19.
  */
class ProbeTask(private val operatorChain: Array[BaseOperator], var data: DataFrame) {
  //savepoint，保存一些运算比较耗时的操作，方便后面的任务可以重用，第一个是算子在算子序列中的索引，第二个是trainData，第三个是testData
  private val savepoint: ArrayBuffer[(Int, DataFrame, DataFrame)] = new ArrayBuffer[(Int, DataFrame, DataFrame)]
  //训练器
  private var trainer: TrainBase = _
  //最终的融合验证值
  private var finalValidation = 0.0

  var trainData: DataFrame = _
  var testData: DataFrame = _
  var runPoint = 0

  /**
    * 获取算子序列
    *
    * @return 算子序列
    */
  def getOperatorChain = this.operatorChain

  def getSavepoint = this.savepoint

  /**
    * 获取当前任务的超参数数组
    *
    * @return 前任务的超参数数组
    */
  def getParams: Array[Double] = operatorChain.flatMap {
    operator =>
      for (i <- 0 until operator.getParamNum) yield operator.getCurrentParam(i)
  }

  /**
    * 更新当前任务所有任务的超参数
    *
    * @param params 要更新的超参数数组
    */
  def updateParams(params: Array[Double]) {
    var right = 0
    operatorChain.foreach {
      operator =>
        val left = right
        right += operator.getParamNum
        operator.updateParam(params.slice(left, right))
    }
  }

  /**
    * 根据超参数在整个超参数序列中的索引获取任务运行点（实际上就是算子在算子序列中的索引）
    *
    * @param paramIndex 超参数在整个超参数序列中的索引
    * @return 算子在算子序列中的索引
    */
  def getRunPoint(paramIndex: Int): Int = {
    var runPoint = 0
    var paramPos = paramIndex
    while (runPoint < operatorChain.length && paramPos >= 0) {
      paramPos -= operatorChain(runPoint).getParamNum
      runPoint += 1
    }
    runPoint - 1
  }

  /**
    * 根据任务运行点（实际上就是算子在算子序列中的索引），获取该算子的超参数在整个超参数序列中的索引范围
    *
    * @param runPoint 算子在算子序列中的索引
    * @return 获取该算子的超参数在整个超参数序列中的索引范围，值为(left,right)
    */
  def getParamIndexRange(runPoint: Int): (Int, Int) = {
    var left = 0
    for (i <- 0 until runPoint) left += operatorChain(i).getParamNum
    (left, left + operatorChain(runPoint).getParamNum - 1)
  }

  /**
    * 获取所有的数据评估值，数据评估值是指，每个算子运算后对数据造成的影响的一种评估
    *
    * @return 所有的数据评估值
    */
  def getEvaluations: Array[Double] = operatorChain.flatMap {
    case operator: EvaluationBase => operator.getEvaluations
    case _ => None
  }

  /**
    * 获取所有的训练效果验证值，可能有auc，ks，bad rate等等多个评估指标
    *
    * @return 所有的训练效果验证值
    */
  def getValidations: Array[(Double, Double)] = operatorChain.flatMap {
    case operator: ValidationBase => operator.getValidations
    case _ => None
  }

  /**
    * 计算最终的训练效果融合验证值
    *
    * @return 最终的训练效果融合验证值
    */
  def calcFinalValidation: Double = {
    val validation = AssemblyValidation.assembleValidation(this.getValidations)
    this.finalValidation = AssemblyValidation.assembleTrainingAndTrainedValidation(trainer.getValidation, validation)
    this.finalValidation
  }

  /**
    * 获取最终的训练效果融合验证值
    *
    * @return 最终的训练效果融合验证值
    */
  def getFinalValidation: Double = this.finalValidation

  /**
    * 获取当前算子序列中的模型训练算子
    *
    * @return 当前算子序列中的模型训练算子
    */
  def getTrainer: TrainBase = {
    if (null == trainer)
      this.trainer = operatorChain(operatorChain.lastIndexWhere(_.isInstanceOf[TrainBase])).asInstanceOf[TrainBase]
    this.trainer
  }
}
