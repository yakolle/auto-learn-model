package org.automl.model.strategy.learn

import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.sql.DataFrame
import org.automl.model.context.ContextHolder

/**
  * Created by zhangyikuo on 2017/3/17.
  */
class GBTRegressionLearner extends LearnerBase {
  private val maxIterations = 10

  private var model: GBTRegressionModel = _


  /**
    * 根据学习过程中的超参数数据进行参数评估模型的学习
    *
    * @param paramData 学习过程中的超参数数据
    */
  override def learn(paramData: DataFrame) {
    val model = new GBTRegressor().setLossType("squared").setMaxIter(maxIterations).fit(paramData)
    this.synchronized(this.model = model)
  }

  /**
    * 根据超参数数组进行超参数评估模型的在线学习
    *
    * @param paramArray 超参数数组，包括目标评估值
    */
  override def onlineLearn(paramArray: Array[Double]) {
    if (null == model) {
      //spark bug，GBTRegressor不支持一条数据
      learn(ContextHolder.toDF(Array(paramArray, paramArray.map(_ + 0.01))))
    }
  }

  /**
    * 预测超参数集合的得分
    *
    * @param paramArray 要评估的超参数数组，不包括目标评估值
    * @return 超参数数组的得分
    */
  override def predict(paramArray: Array[Double]): Double = {
    val params = ContextHolder.toDF(Array(paramArray :+ 0.0))
    var predictData: DataFrame = null
    this.synchronized(predictData = model.transform(params))
    predictData.select("prediction").head.getDouble(0)
  }

  /**
    * 预测超参数集合的得分
    *
    * @param params 要评估的超参数集合
    * @return 超参数集合的预测得分，得分列名为"prediction"
    */
  override def predict(params: DataFrame): DataFrame = this.synchronized(model.transform(params))

  /**
    * 获取各超参数的权重
    *
    * @return 各超参数的权重
    */
  override def getWeights: Array[Double] = Array.fill(model.numFeatures)(0.0)


  /**
    * 获取各超参数重要程度评估，而超参数的权重有可能表示重要程度，但也有可能只是系数而已，取决于使用哪种学习器
    *
    * @return 各超参数重要程度评估，各子类要保证重要程度都是非负的，并且是归一化的
    */
  override def getParamImportances: Array[Double] = {
    val importances = model.featureImportances.toArray
    val sum = importances.sum
    if (sum <= 0.0) importances else importances.map(_ / sum)
  }

  /**
    * clone方法
    *
    * @return 一个当前学习器的副本
    */
  override def clone: LearnerBase = super.clone.asInstanceOf[GBTRegressionLearner]
}
