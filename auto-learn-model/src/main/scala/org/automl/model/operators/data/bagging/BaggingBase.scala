package org.automl.model.operators.data.bagging

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2016/8/19.
  */
abstract class BaggingBase extends BaseOperator {
  this.operatorName = "bagging"
  this.operatorType = "bagging"
  this.procedureType = "bagging"

  /**
    * 运行数据拆分算子，拆分前后都要对原始数据进行随机打乱
    *
    * @param data    数据（包含X,y）
    * @param abRatio 要拆分的两部分数据的比例
    * @return A、B两部分数据，返回值为((aX,ay),(bX,by))
    */
  def run(data: DataFrame, abRatio: Double = 1): (DataFrame, DataFrame)
}
