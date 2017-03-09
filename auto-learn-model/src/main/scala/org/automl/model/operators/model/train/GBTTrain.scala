package org.automl.model.operators.model.train

import java.io.{BufferedWriter, IOException}

import org.apache.spark.ml.classification.{GBTClassificationModel, GBTClassifier}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.operators.model.validation.{AssemblyValidation, ValidationBase}

/**
  * Created by zhangyikuo on 2017/3/9.
  */
class GBTTrain(kFold: Int = 5) extends TrainBase {
  this.operatorName = "gbt"

  //上次cv验证值
  private var validation = 0.0
  private var model: GBTClassificationModel = _
  //依次为：maxIterations,maxBins,maxDepth,minInstancesPerNode,stepSize,subsamplingRate
  private var params = Array(50.0, 32.0, 5.0, 1.0, 0.1, 1.0)

  //warm start
  private val empiricalParams = Array(50.0, 32.0, 5.0, 1.0, 0.1, 1.0)
  //各参数的搜索范围
  private val paramBoundaries = Array((1.0, 100.0), (2.0, 1024.0), (1.0, 10.0), (1.0, 30.0), (1E-4, 1.0), (0.5, 1.0))
  //各参数的经验搜索步幅
  private val empiricalParamPaces = Array(1.0, 1.0, 1.0, 1.0, 1E-4, 0.02)
  //各参数类型
  private val paramTypes = Array(BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_INT,
    BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_DOUBLE, BaseOperator.PARAM_TYPE_DOUBLE)

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
  override def run(data: DataFrame): (Double, GBTClassificationModel) = {
    val trainer = new GBTClassifier().setMaxIter(params(0).toInt).setMaxBins(params(1).toInt).setMaxDepth(params(2).toInt)
      .setMinInstancesPerNode(params(3).toInt).setStepSize(params(4)).setSubsamplingRate(params(5))
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
  override def getModel: GBTClassificationModel = this.model

  /**
    * 输出训练得到的model的主要参数，以便别的程序（不支持spark的程序）可以利用这些参数，重新构建练得到的model
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  override def explainModel(out: BufferedWriter) {
    out.write(model.toDebugString)
    out.flush()
  }

  /**
    * 获取上次运行（调用run方法）后cv验证值
    *
    * @return 上次训练后cv验证值
    */
  override def getValidation: Double = this.validation

  /**
    * 获取超参数个数
    *
    * @return 超参数个数，如果没有超参数，返回0
    */
  override def getParamNum: Int = empiricalParams.length

  /**
    * 获取超参数的搜索范围
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的搜索范围，返回值为(minParam,maxParam)
    */
  override def getParamBoundary(data: DataFrame, paramIndex: Int): (Double, Double) = paramBoundaries(paramIndex)

  /**
    * 获取超参数的类型，只有三种，double、int、boolean，默认为double
    *
    * @param paramIndex 第几个（从0开始）超参数
    * @return 超参数的类型
    */
  override def getParamType(paramIndex: Int): Int = paramTypes(paramIndex)

  /**
    * 获取超参数的经验搜索步幅
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验搜索步幅
    */
  override def getEmpiricalParamPace(data: DataFrame, paramIndex: Int): Double = empiricalParamPaces(paramIndex)

  /**
    * 获取warm start点（超参数经验搜索起始点）
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验值（warm start点）
    */
  override def getEmpiricalParam(data: DataFrame, paramIndex: Int): Double = empiricalParams(paramIndex)

  /**
    * 获取超参数当前值
    *
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的当前值
    */
  override def getCurrentParam(paramIndex: Int): Double = params(paramIndex)

  /**
    * 更新超参数，如果算子不需要参数可以不用重写该方法，否则必须重写该方法
    *
    * @param params 要更新的超参数
    */
  override def updateParam(params: Array[Double]) {
    this.params = params
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = {
    val copy = super.clone.asInstanceOf[GBTTrain]
    copy.validators = this.validators.map(_.clone.asInstanceOf[ValidationBase])
    copy
  }
}
