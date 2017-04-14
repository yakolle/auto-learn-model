package org.automl.model.strategy.learn

import org.apache.spark.sql.DataFrame

/**
  * Created by zhangyikuo on 2016/8/24.
  */
abstract class LearnerBase extends Cloneable {
  /**
    * 根据学习过程中的超参数数据进行参数评估模型的学习
    *
    * @param paramData 学习过程中的超参数数据
    */
  def learn(paramData: DataFrame)

  /**
    * 根据超参数数组进行超参数评估模型的在线学习
    *
    * @param paramArray 超参数数组，包括目标评估值
    */
  def onlineLearn(paramArray: Array[Double])

  /**
    * 预测超参数集合的得分
    *
    * @param paramArray 要评估的超参数数组，不包括目标评估值
    * @return 超参数数组的得分
    */
  def predict(paramArray: Array[Double]): Double

  /**
    * 预测超参数集合的得分
    *
    * @param params 要评估的超参数集合
    * @return 超参数集合的预测得分，得分列名为"prediction"
    */
  def predict(params: DataFrame): DataFrame

  /**
    * 获取各超参数重要程度评估，而超参数的权重有可能表示重要程度，但也有可能只是系数而已，取决于使用哪种学习器
    *
    * @return 各超参数重要程度评估，各子类要保证重要程度都是非负的，并且是归一化的
    */
  def getParamImportances: Array[Double] = getWeights

  /**
    * 获取各超参数的权重
    *
    * @return 各超参数的权重
    */
  def getWeights: Array[Double]

  /**
    * clone方法
    *
    * @return 一个当前学习器的副本
    */
  override def clone: LearnerBase = super.clone.asInstanceOf[LearnerBase]
}
