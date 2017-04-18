package org.automl.model.operators.model.train

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql._
import org.automl.model.operators.BaseOperator
import org.automl.model.operators.model.validation.{AssemblyValidation, ValidationBase}

/**
  * Created by zhangyikuo on 2017/3/10.
  */
class RFTrain(kFold: Int = 5) extends TrainBase {
  this.operatorName = "rf"

  /*
  依次为：featureSubsetStrategy("auto","all","onethird","sqrt","log2","n"),nFeatureSubsetStrategy(featureSubsetStrategy中"n"策略对应值),
  impurity(1为"gini"，0为"entropy"),maxBins,maxDepth,minInstancesPerNode,numTrees,subsamplingRate
   */
  this.params = Array(0.0, 0.0, 1.0, 32.0, 5.0, 1.0, 20.0, 1.0)

  this.empiricalParams = Array(0.0, 0.0, 1.0, 32.0, 5.0, 1.0, 20.0, 1.0)
  this.paramBoundaries = Array((0.0, 5.0), (0.05, 1.0), (0.0, 1.0), (2.0, 1024.0), (1.0, 10.0), (1.0, 30.0), (1.0, 200.0), (0.5, 1.0))
  this.empiricalParamPaces = Array(1.0, 0.05, 0.5, 1.0, 1.0, 1.0, 1.0, 0.02)
  this.paramTypes = Array(BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_DOUBLE, BaseOperator.PARAM_TYPE_BOOLEAN,
    BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_INT,
    BaseOperator.PARAM_TYPE_DOUBLE)

  //上次cv验证值
  private var validation = 0.0
  private var model: RandomForestClassificationModel = _

  //训练过程中验证算子列表
  private var validators: Array[ValidationBase] = _

  def setValidators(validators: Array[ValidationBase]) {
    this.validators = validators
  }

  /**
    * 运行模型训练算子
    *
    * @param data 数据（包含X,y）
    * @return 本次训练完成后的模型及cv验证值，返回值为(cv验证值,模型)
    */
  override def run(data: DataFrame): (Double, RandomForestClassificationModel) = {
    val trainer = new RandomForestClassifier().setImpurity(if (1.0 == params(2)) "gini" else "entropy").setMaxBins(params(3).toInt)
      .setMaxDepth(params(4).toInt).setMinInstancesPerNode(params(5).toInt).setNumTrees(params(6).toInt).setSubsamplingRate(params(7))
    params(0) match {
      case 0.0 => trainer.setFeatureSubsetStrategy("auto")
      case 1.0 => trainer.setFeatureSubsetStrategy("all")
      case 2.0 => trainer.setFeatureSubsetStrategy("onethird")
      case 3.0 => trainer.setFeatureSubsetStrategy("sqrt")
      case 4.0 => trainer.setFeatureSubsetStrategy("log2")
      case _ => trainer.setFeatureSubsetStrategy(params(1).toString)
    }
    var totalValidations = 0.0

    MLUtils.kFold(data.rdd, kFold, System.currentTimeMillis).foreach {
      dataPair =>
        val trainData = data.sparkSession.createDataFrame(dataPair._1, data.schema)
        val testData = data.sparkSession.createDataFrame(dataPair._2, data.schema)
        val model = trainer.fit(trainData)
        totalValidations += AssemblyValidation.assembleTrainValidation(this.validators.flatMap {
          validator =>
            validator.run(trainData, model, testData)
            validator.getValidations
        })
    }

    this.validation = totalValidations / kFold
    this.model = trainer.fit(data)
    (this.validation, this.model)
  }

  /**
    * 返回上次训练后得到的模型
    *
    * @return 上次训练后得到的模型
    */
  override def getModel: RandomForestClassificationModel = this.model

  /**
    * 获取上次运行（调用run方法）后cv验证值
    *
    * @return 上次训练后cv验证值
    */
  override def getValidation: Double = this.validation

  /**
    * 格式化超参数，如果算子不需要参数可以不用重写该方法，否则必须重写该方法
    *
    * @param params 需要格式化的超参数
    * @return 格式化后的超参数
    */
  override protected def formatParamInternal(params: Array[Double]): Array[Double] = {
    if (params(0) < 5.0) params.updated(1, 0.0) else params
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = {
    val copy = super.clone.asInstanceOf[RFTrain]
    copy.validators = this.validators.map(_.clone.asInstanceOf[ValidationBase])
    copy
  }
}
