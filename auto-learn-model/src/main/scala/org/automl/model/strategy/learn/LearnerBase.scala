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
    * 获取各超参数的权重
    *
    * @return 各超参数的权重
    */
  def getWeights: Array[Double]

  /**
    * 获取各超参数的权重，以及一些必要参数，比如线性回归的截距项
    *
    * @return 各超参数的权重，及一些必要参数
    */
  def getAllWeights: Array[Double]

  /**
    * clone方法
    *
    * @return 一个当前学习器的副本
    */
  override def clone: LearnerBase = super.clone.asInstanceOf[LearnerBase]
}
