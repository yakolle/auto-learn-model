package org.automl.model.operators.model.train

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.operators.model.validation.{AssemblyValidation, ValidationBase}

/**
  * Created by zhangyikuo on 2017/1/11.
  */
class LogisticRegressionTrain(kFold: Int = 5) extends TrainBase {
  this.operatorName = "lr"

  //just lamda
  this.params = Array(0.2)

  this.empiricalParams = Array(0.2)
  this.paramBoundaries = Array((0.0, 100.0))
  this.empiricalParamPaces = Array(0.01)

  private val maxIterations = 100

  //上次cv验证值
  private var validation = 0.0
  private var model: LogisticRegressionModel = _

  //训练过程中验证算子列表
  private var validators: Array[ValidationBase] = _

  def setValidators(validators: Array[ValidationBase]) {
    this.validators = validators
  }

  /**
    * 运行模型训练算子
    *
    * @param data 数据（包含X,y），其中X为Vector[Double]类型
    * @return 本次训练完成后的模型及cv验证值，返回值为(cv验证值,模型)
    */
  override def run(data: DataFrame): (Double, LogisticRegressionModel) = {
    val trainer = new LogisticRegression().setMaxIter(maxIterations).setRegParam(params(0)).setElasticNetParam(0.0)
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
  override def getModel: LogisticRegressionModel = this.model

  /**
    * 获取上次运行（调用run方法）后cv验证值
    *
    * @return 上次训练后cv验证值
    */
  override def getValidation: Double = this.validation

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = {
    val copy = super.clone.asInstanceOf[LogisticRegressionTrain]
    copy.validators = this.validators.map(_.clone.asInstanceOf[ValidationBase])
    copy
  }
}
