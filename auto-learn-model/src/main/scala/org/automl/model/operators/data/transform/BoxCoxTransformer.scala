package org.automl.model.operators.data.transform

import java.io.{BufferedWriter, IOException}

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2017/6/6.
  */
class BoxCoxTransformer extends TransformBase {
  this.operatorName = "boxcox"
  this.procedureType = "preprocess"

  //lamda值一共为17种，-4, -3.5, -3, ..., 3, 3.5, 4
  private val types: Int = 17
  private val logTypeIndex: Int = types / 2

  //依次为：isOn,lamda
  this.params = Array(1.0, logTypeIndex)

  this.empiricalParams = Array(1.0, logTypeIndex)
  this.paramBoundaries = Array((0.0, 1.0), (0.0, types * 1E6))
  this.empiricalParamPaces = Array(0.5, 1.0)
  this.paramTypes = Array(BaseOperator.PARAM_TYPE_BOOLEAN, BaseOperator.PARAM_TYPE_INT)

  private var featuresNum = Int.MaxValue

  /**
    * 运行数据处理算子
    *
    * @param data 数据（包含X,y），其中X并非Vector[Double]类型，其中的每个一x为单独的一列
    * @return 经过处理后的数据
    */
  override def run(data: DataFrame): DataFrame = {
    if (Int.MaxValue == featuresNum) {
      featuresNum = data.columns.length - 1
      val upper = types * featuresNum.toDouble - 1.0
      paramBoundaries(1) = (0.0, upper)
    }

    transform(data)
  }

  /**
    * 对数据进行transform
    *
    * @param data 数据（包含X,y），其中X并非Vector[Double]类型，其中的每个一x为单独的一列
    * @return transform后的数据
    */
  override def transform(data: DataFrame): DataFrame = {
    val feature = data.columns.filter(_ != "label")((params(1) / featuresNum).toInt)
    val lamda = params(1).toInt % featuresNum

    val transUDF = if (logTypeIndex == lamda) udf((value: Double) => if (value <= 0) 0.0 else math.log1p(value))
    else udf((value: Double) => math.pow(value, 0.5 * (lamda - logTypeIndex)))

    data.withColumn(feature + "_boxcox_" + lamda, transUDF(col(feature))).drop(feature)
  }

  /**
    * 输出transformer的主要属性，以便别的程序可以利用这些属性，对新数据进行transform
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  override def explain(out: BufferedWriter) {
    out.write((params(1) / featuresNum).toInt.toString)
    out.write((0.5 * (params(1).toInt % featuresNum - logTypeIndex)).toString)
    out.flush()
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[BoxCoxTransformer]
}
