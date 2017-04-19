package org.automl.model.operators.data.expand

import java.io.{BufferedWriter, IOException}

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2017/4/20.
  */
abstract class ExpandFeaturesBase extends BaseOperator {
  this.operatorName = "expand"
  this.operatorType = "expand"
  this.procedureType = "expand"

  /**
    * 运行特征扩展算子
    *
    * @param data 数据（包含X,y）
    * @return 特征扩展后的数据
    */
  def run(data: DataFrame): DataFrame

  /**
    * 对数据进行特征扩展
    *
    * @param data 数据（包含X,y）
    * @return 特征扩展后的数据
    */
  def transform(data: DataFrame): DataFrame

  /**
    * 输出expandor的主要属性，以便别的程序可以利用这些属性，对新数据进行expand
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  @throws(classOf[IOException])
  def explain(out: BufferedWriter)
}