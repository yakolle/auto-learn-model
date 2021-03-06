package org.automl.model.strategy.learn

import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
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

  private var model: LinearRegressionModel = _
  //系数向量
  private var coefficients: Array[Double] = _
  //截距项
  private var intercept: Double = 0.0

  //归一化后的系数向量
  private var normalizedCoefficients: Array[Double] = _

  /**
    * 根据学习过程中的超参数数据进行参数评估模型的学习
    *
    * @param params 学习过程中的超参数数据，所有超参数assemble为Vector[Double]类型，验证值为其label
    */
  override def learn(params: DataFrame) {
    val model = new LinearRegression().setMaxIter(maxIterations).setRegParam(lambda).setElasticNetParam(elasticNetParam).fit(params)

    val nzParams = new MinMaxScaler().setInputCol("features").setOutputCol("transformedFeatures").fit(params).transform(params)
    val nzModel = new LinearRegression().setMaxIter(maxIterations).setRegParam(lambda).setElasticNetParam(elasticNetParam)
      .setFeaturesCol("transformedFeatures").fit(nzParams)

    this.synchronized {
      this.model = model
      coefficients = model.coefficients.toArray
      intercept = model.intercept

      normalizedCoefficients = nzModel.coefficients.toArray
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
        normalizedCoefficients = Array.fill(x.length)(0.0)
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
    * 预测超参数集合的得分
    *
    * @param params 要评估的超参数集合
    * @return 超参数集合的预测得分，得分列名为"prediction"
    */
  override def predict(params: DataFrame): DataFrame = this.synchronized(model.transform(params))

  /**
    * 获取各超参数重要程度评估，而超参数的权重有可能表示重要程度，但也有可能只是系数而已，取决于使用哪种学习器
    *
    * @return 各超参数重要程度评估，各子类要保证重要程度都是非负的，并且是归一化的
    */
  override def getParamImportances: Array[Double] = {
    val importances = normalizedCoefficients.map(math.abs)
    val sum = importances.sum
    if (sum <= 0.0) importances else importances.map(_ / sum)
  }

  /**
    * 获取各超参数的权重
    *
    * @return 各超参数的权重
    */
  override def getWeights: Array[Double] = coefficients

  /**
    * clone方法
    *
    * @return 一个当前学习器的副本
    */
  override def clone: LearnerBase = {
    val copy = super.clone.asInstanceOf[LinearRegressionLearner]
    copy.coefficients = this.coefficients.clone
    copy.normalizedCoefficients = this.normalizedCoefficients.clone
    copy
  }
}
