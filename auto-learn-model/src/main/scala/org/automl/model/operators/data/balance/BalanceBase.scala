package org.automl.model.operators.data.balance

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2016/8/19.
  */
abstract class BalanceBase extends BaseOperator {
  this.operatorName = "balance"
  this.operatorType = "balance"
  this.procedureType = "balance"

  /**
    * 运行数据均衡算子，均衡后要对数据进行随机打乱
    *
    * @param data 数据（包含X,y）
    * @return 均衡后数据
    */
  def run(data: DataFrame): DataFrame
}
