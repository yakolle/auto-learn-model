package org.automl.model.strategy.learn

import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}
import org.automl.model.utils.MathUtil

/**
  * Created by zhangyikuo on 2017/3/17.
  */
class AdaptableLearner extends LearnerBase {
  private var learnerArray = Array[LearnerBase](new LinearRegressionLearner, new GBTRegressionLearner)
  private var learnerWeights = Array.fill(learnerArray.length)(1.0 / learnerArray.length.toDouble)

  /**
    * 根据学习过程中的超参数数据进行参数评估模型的学习
    *
    * @param paramData 学习过程中的超参数数据
    */
  override def learn(paramData: DataFrame) {
    learnerArray.foreach(_.learn(paramData))

    val errs = learnerArray.map { learner =>
      learner.predict(paramData).agg(sum(abs(col("prediction") - col("label")))).head.getDouble(0)
    }

    //避免某些学习器的权重变成了0，即使有个学习器没有错误
    val minMaxErr = errs.max * (1.0 + 1E-3) + errs.min
    val errSum = minMaxErr * errs.length - errs.sum
    val learnWeights = errs.map(err => (minMaxErr - err) / errSum)

    this.synchronized(this.learnerWeights = learnWeights)
  }

  /**
    * 根据超参数数组进行超参数评估模型的在线学习
    *
    * @param paramArray 超参数数组，包括目标评估值
    */
  override def onlineLearn(paramArray: Array[Double]) {
    learnerArray.foreach(_.onlineLearn(paramArray))
  }

  /**
    * 预测超参数集合的得分
    *
    * @param paramArray 要评估的超参数数组，不包括目标评估值
    * @return 超参数数组的得分
    */
  override def predict(paramArray: Array[Double]): Double = MathUtil.dot(learnerArray.map(_.predict(paramArray)), learnerWeights)

  /**
    * 预测超参数集合的得分
    *
    * @param params 要评估的超参数集合
    * @return 超参数集合的预测得分，得分列名为"prediction"
    */
  override def predict(params: DataFrame): DataFrame = {
    import scala.collection.JavaConverters._

    val predict = params.collect().map { row =>
      Row.fromSeq(Seq(this.predict(row.getAs[Vector]("features").toArray)))
    }.toSeq.asJava

    params.sparkSession.createDataFrame(predict, StructType(Array(StructField("prediction", DoubleType, nullable = false))))
  }

  /**
    * 按照learnerWeights合并权重
    *
    * @param weightMatrix 权重矩阵
    * @return 合并后的权重
    */
  private def mergeWeights(weightMatrix: Array[Array[Double]]): Array[Double] = {
    val weightLen = weightMatrix.head.length
    val mergedWeights = Array.fill(weightLen)(0.0)
    for (i <- weightMatrix.indices) {
      val learnerWeight = learnerWeights(i)
      val weights = weightMatrix(i)
      for (j <- weights.indices) mergedWeights(j) += learnerWeight * weights(j)
    }
    mergedWeights
  }

  /**
    * 获取各超参数的权重
    *
    * @return 各超参数的权重
    */
  override def getWeights: Array[Double] = mergeWeights(learnerArray.map(_.getWeights))

  /**
    * 获取各超参数重要程度评估，而超参数的权重有可能表示重要程度，但也有可能只是系数而已，取决于使用哪种学习器
    *
    * @return 各超参数重要程度评估，各子类要保证重要程度都是非负的，并且是归一化的
    */
  override def getParamImportances: Array[Double] = mergeWeights(learnerArray.map(_.getParamImportances))

  /**
    * clone方法
    *
    * @return 一个当前学习器的副本
    */
  override def clone: LearnerBase = {
    val copy = super.clone.asInstanceOf[AdaptableLearner]
    copy.learnerArray = learnerArray.map(_.clone)
    copy
  }
}
