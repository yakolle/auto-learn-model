package org.automl.model.operators.data.transform

import java.io.{BufferedWriter, IOException}

import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2017/1/17.
  */
class MinMaxMapper extends TransformBase {
  this.operatorName = "minmax"
  this.procedureType = "preprocess"

  private var transModel: MinMaxScalerModel = _

  /**
    * 运行数据处理算子
    *
    * @param data 数据（包含X,y），其中X为Vector[Double]类型
    * @return 经过处理后的数据
    */
  override def run(data: DataFrame): DataFrame = {
    this.transModel = new MinMaxScaler().setInputCol("features").setOutputCol("transformedFeatures").fit(data)
    transform(data)
  }

  /**
    * 对数据进行transform
    *
    * @param data 数据（包含X,y），其中X为Vector[Double]类型
    * @return transform后的数据
    */
  override def transform(data: DataFrame): DataFrame = {
    val transformedData = this.transModel.transform(data)
    val schema = StructType(Array(transformedData.schema("label"), transformedData.schema("features")))
    transformedData.sparkSession.createDataFrame(transformedData.drop("features").rdd, schema).select("features", "label")
  }

  /**
    * 输出transformer的主要属性，以便别的程序可以利用这些属性，对新数据进行transform
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  override def explain(out: BufferedWriter) {
    val mins = transModel.originalMin.toArray
    val maxes = transModel.originalMax.toArray

    for (i <- mins.indices) {
      out.write(mins(i).toString)
      out.write("\t")
      out.write(maxes(i).toString)
      out.newLine()
    }
    out.flush()
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[MinMaxMapper]
}
