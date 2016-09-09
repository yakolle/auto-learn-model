package org.automl.model.operators.data.format

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.utils.DataTransformUtil

/**
  * Created by zhangyikuo on 2017/2/9.
  */
class DataAssembler extends FormatBase {
  this.operatorName = "assemble"
  this.procedureType = "preprocess"

  /**
    * 运行数据格式化算子，将数据格式化为后面数据处理流程所需格式
    *
    * @param data 原数据
    * @return 格式化后的数据
    */
  override def run(data: DataFrame): DataFrame = DataTransformUtil.dataSchemaTransform(data)

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[DataAssembler]
}
