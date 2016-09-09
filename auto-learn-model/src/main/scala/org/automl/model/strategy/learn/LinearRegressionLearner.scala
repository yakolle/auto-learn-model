package org.automl.model.strategy.learn

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.DataFrame
import org.automl.model.utils.MathUtil

/**
  * Created by zhangyikuo on 2016/8/24.
  */
class LinearRegressionLearner extends LearnerBase {
  //在线学习的学习率
  private val onlineLearningRate = 0.001
  private val maxIterations = 100
  private val lambda = 0.1
  private val elasticNetParam = 0.5

  //系数向量
  private var coefficients: Array[Double] = _
  //截距项
  private var intercept: Double = 0.0

  /**
    * 根据学习过程中的超参数数据进行参数评估模型的学习
    *
    * @param params 学习过程中的超参数数据，所有超参数assemble为Vector[Double]类型，验证值为其label
    */
  override def learn(params: DataFrame) {
    val lmModel = new LinearRegression().setMaxIter(maxIterations).setRegParam(lambda).setElasticNetParam(elasticNetParam).fit(params)
    this.synchronized {
      coefficients = lmModel.coefficients.toArray
      intercept = lmModel.intercept
    }
  }

  /**
    * 根据超参数集合进行超参数评估模型的在线学习
    *
    * @param param 超参数集合，验证值位于数组末尾
    */
  override def onlineLearn(param: Array[Double]) {
    val x = param.dropRight(1)

    this.synchronized {
      val stepUnit = if (null == coefficients) {
        coefficients = Array.fill(x.length)(0.0)
        -param.last
      } else onlineLearningRate * (predict(x) - param.last)

      for (i <- coefficients.indices) coefficients(i) -= x(i) * stepUnit
      this.intercept -= stepUnit
    }
  }

  /**
    * 预测超参数集合的得分
    *
    * @param param 要评估的超参数集合
    * @return 超参数集合的得分
    */
  override def predict(param: Array[Double]): Double = MathUtil.dot(coefficients, param) + intercept

  /**
    * 获取各超参数的权重
    *
    * @return 各超参数的权重
    */
  override def getWeights: Array[Double] = coefficients

  /**
    * 获取各超参数的权重，以及一些必要参数，比如线性回归的截距项
    *
    * @return 各超参数的权重，及一些必要参数
    */
  override def getAllWeights: Array[Double] = intercept +: coefficients

  /**
    * clone方法
    *
    * @return 一个当前学习器的副本
    */
  override def clone: LearnerBase = {
    val copy = super.clone.asInstanceOf[LinearRegressionLearner]
    copy.coefficients = this.coefficients.clone
    copy
  }
}
