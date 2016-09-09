package org.automl.model.operators.data.bagging

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2016/10/12.
  */
class ABBagging extends BaggingBase {
  /**
    * 运行数据拆分算子，拆分前后都要对原始数据进行随机打乱
    *
    * @param data    数据（包含X,y）
    * @param abRatio 要拆分的两部分数据的比例
    * @return A、B两部分数据，返回值为((aX,ay),(bX,by))
    */
  override def run(data: DataFrame, abRatio: Double): (DataFrame, DataFrame) = {
    val Array(train, test) = data.randomSplit(Array(abRatio / (abRatio + 1), 1 / (abRatio + 1)))
    (train, test)
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[ABBagging]
}
