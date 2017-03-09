package org.automl.model.operators.model.validation

import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2017/1/10.
  */
class AUCValidation extends ValidationBase {
  this.operatorName = "auc"

  //格式为Array(trainAUC,testAUC)
  protected var aucArray: Array[(Double, Double)] = new Array[(Double, Double)](1)

  /**
    * 运行AUC模型验证算子，评估数据（包含X,y）共两列，X为features(一列)，y为target（一列），并且每一行的features为
    * 一个assembledVector，并非每个feature为一列，如有需要可考虑用DataTransformUtil.dataSchemaTransform进行transform
    *
    * @param trainData 训练数据（包含X,y）
    * @param model     模型
    * @param testData  测试数据（包含X,y）
    * @return 本次模型AUC验证得分数组，格式为Array(trainAUC,testAUC)
    */
  override def run(trainData: DataFrame, model: PredictionModel[_, _], testData: DataFrame): Array[(Double, Double)] = {
    aucArray(0) = (if (null != trainData) AUCValidation.calcAUC(trainData, model) else 0.0,
      if (null != testData) AUCValidation.calcAUC(testData, model) else 0.0)
    this.aucArray
  }

  /**
    * 获取上次模型验证得分数组
    *
    * @return 上次模型验证得分数组
    */
  override def getValidations: Array[(Double, Double)] = this.aucArray


  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = {
    val clone = super.clone.asInstanceOf[AUCValidation]
    clone.aucArray = this.aucArray.clone
    clone
  }
}

object AUCValidation {
  /**
    * 预测得分
    *
    * @param data  评估数据（包含X,y）共两列，X为features(一列)，y为target（一列），并且每一行的features为一个assembledVector，
    *              并非每个feature为一列，可考虑用DataTransformUtil.dataSchemaTransform进行transform
    * @param model 预测模型
    * @return 由model得出的预测得分
    */
  def predict(data: DataFrame, model: PredictionModel[_, _]): RDD[(Double, Double)] = {
    model.transform(data).select("probability", "label").rdd.map { row =>
      (row.getAs[DenseVector](0)(1), row(1).toString.toDouble)
    }
  }

  /**
    * 计算AUC
    *
    * @param data  评估数据（包含X,y）共两列，X为features(一列)，y为target（一列），并且每一行的features为一个assembledVector，
    *              并非每个feature为一列，可考虑用DataTransformUtil.dataSchemaTransform进行transform
    * @param model 预测模型
    * @return auc
    */
  def calcAUC(data: DataFrame, model: PredictionModel[_, _]) = new BinaryClassificationMetrics(predict(data, model))
    .areaUnderROC
}


