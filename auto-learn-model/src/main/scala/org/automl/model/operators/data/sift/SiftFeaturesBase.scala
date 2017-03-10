package org.automl.model.operators.data.sift

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.utils.DataTransformUtil

/**
  * Created by zhangyikuo on 2016/8/19.
  */
abstract class SiftFeaturesBase extends BaseOperator {
  this.operatorName = "sift"
  this.operatorType = "sift"
  this.procedureType = "sift"

  /**
    * 运行特征筛选算子
    *
    * @param data 数据（包含X,y）
    * @return 筛选后的数据及特征，返回值为筛选后的数据
    */
  def run(data: DataFrame): DataFrame

  /**
    * 对数据进行特征筛选
    *
    * @param data 数据（包含X,y）
    * @return 进过特征筛选后的数据
    */
  def transform(data: DataFrame): DataFrame = DataTransformUtil.selectFeaturesFromAssembledData(data, this.getFeatureIDs)

  /**
    * 获取该算子目前所选择的特征ID数组
    *
    * @return 算子目前所选择的特征ID数组
    */
  def getFeatureIDs: Array[String]
}
