package org.automl.model.operators.data.format

import org.apache.spark.sql._
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2017/2/9.
  */
abstract class FormatBase extends BaseOperator {
  this.operatorName = "format"
  this.operatorType = "format"
  this.procedureType = "format"

  /**
    * 运行数据格式化算子，将数据格式化为后面数据处理流程所需格式
    *
    * @param data 原数据
    * @return 格式化后的数据
    */
  def run(data: DataFrame): DataFrame
}
