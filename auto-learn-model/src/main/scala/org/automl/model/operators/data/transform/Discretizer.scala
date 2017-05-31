package org.automl.model.operators.data.transform

import java.io.{BufferedWriter, IOException}

import org.apache.spark.ml.feature.{Bucketizer, QuantileDiscretizer}
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2017/5/31.
  */
class Discretizer extends TransformBase {
  this.operatorName = "discretize"
  this.procedureType = "preprocess"

  //依次为：isOn,numBuckets
  this.params = Array(1.0, 2.0)

  this.empiricalParams = Array(1.0, 2.0)
  this.paramBoundaries = Array((0.0, 1.0), (2.0, 1E5))
  this.empiricalParamPaces = Array(0.5, 1.0)
  this.paramTypes = Array(BaseOperator.PARAM_TYPE_BOOLEAN, BaseOperator.PARAM_TYPE_INT)

  private var bucketizers: Array[Bucketizer] = _


  /**
    * 运行数据处理算子
    *
    * @param data 数据（包含X,y），其中X并非Vector[Double]类型，其中的每个一x为单独的一列
    * @return 经过处理后的数据
    */
  override def run(data: DataFrame): DataFrame = {
    bucketizers = data.columns.filter(_ != "label").map { featureName =>
      new QuantileDiscretizer().setInputCol(featureName).setOutputCol(featureName + "_rank").setNumBuckets(params(1).toInt).fit(data)
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
    var transformedData = data
    bucketizers.foreach(bucketizer => transformedData = bucketizer.transform(transformedData))
    transformedData
  }

  /**
    * 输出transformer的主要属性，以便别的程序可以利用这些属性，对新数据进行transform
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  override def explain(out: BufferedWriter) {
    bucketizers.foreach { bucketizer =>
      out.write(bucketizer.getInputCol + ": ")

      bucketizer.getSplits.foreach { bucket =>
        out.write(bucket.toString)
        out.write("\t")
      }

      out.newLine()
    }

    out.flush()
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[Discretizer]
}
