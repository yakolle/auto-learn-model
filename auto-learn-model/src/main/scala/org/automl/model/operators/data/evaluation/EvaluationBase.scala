package org.automl.model.operators.data.evaluation

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * 数据评估算子是在模型训练前，数据操作对数据分布影响的量化评估，比如相关性、信息熵等评价指标
  * Created by zhangyikuo on 2016/8/19.
  */
abstract class EvaluationBase extends BaseOperator {
  this.operatorName = "eval"
  this.operatorType = "eval"
  this.procedureType = "any"

  /**
    * 运行数据评估算子，各算子可能计算同一类型的多个指标，比如计算按0.1-0.9共9个得分为阈值的9个sensitivity值
    *
    * @param data 数据（包含X,y）
    * @return 本次数据评估得分数组
    */
  def run(data: DataFrame): Array[Double]

  /**
    * 获取上次数据评估得分数组
    *
    * @return 上次数据评估得分数组
    */
  def getEvaluations: Array[Double]
}
