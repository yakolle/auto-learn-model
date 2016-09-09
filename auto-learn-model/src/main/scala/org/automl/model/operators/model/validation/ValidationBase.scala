package org.automl.model.operators.model.validation

import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * 模型验证算子，是对训练出的模型，进行一些量化验证，比如AUC、KS、F值等
  * Created by zhangyikuo on 2016/8/19.
  */
abstract class ValidationBase extends BaseOperator {
  this.operatorName = "validation"
  this.operatorType = "validation"
  this.procedureType = "validation"

  /**
    * 运行模型验证算子，各算子可能计算同一类型的多个指标，比如计算训练和测试集上按0.1-0.9共9个得分为阈值的9对sensitivity值
    *
    * @param trainData 训练数据（包含X,y）
    * @param model     模型
    * @param testData  测试数据（包含X,y）
    * @return 本次模型验证得分数组，数组元素为2元组，格式为(trainValidation,testValidation)
    */
  def run(trainData: DataFrame, model: ClassificationModel[_, _], testData: DataFrame): Array[(Double, Double)]

  /**
    * 获取上次模型验证得分数组
    *
    * @return 上次模型验证得分数组，数组元素为2元组，格式为(trainValidation,testValidation)
    */
  def getValidations: Array[(Double, Double)]
}
