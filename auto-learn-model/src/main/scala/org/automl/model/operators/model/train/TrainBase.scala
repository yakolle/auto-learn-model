package org.automl.model.operators.model.train

import org.apache.spark.ml.classification.ClassificationModel
import org.apache.spark.ml.util.MLWritable
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2017/1/11.
  */
abstract class TrainBase extends BaseOperator {
  this.operatorName = "train"
  this.operatorType = "train"
  this.procedureType = "train"

  /**
    * 运行模型训练算子
    *
    * @param data 数据（包含X,y）
    * @return 本次训练完成后的模型及cv验证值，返回值为(cv验证值,模型)
    */
  def run(data: DataFrame, kFold: Int = 5): (Double, ClassificationModel[_, _] with MLWritable)


  /**
    * 返回上次训练后得到的模型
    *
    * @return 上次训练后得到的模型
    */
  def getModel: ClassificationModel[_, _] with MLWritable

  /**
    * 获取上次运行（调用run方法）后cv验证值
    *
    * @return 上次训练后cv验证值
    */
  def getValidation: Double
}
