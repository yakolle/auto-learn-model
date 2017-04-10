package org.automl.model.operators.data.transform

import java.io.{BufferedWriter, IOException}

import org.apache.spark.sql.DataFrame
import org.automl.model.context.ContextHolder
import org.automl.model.operators.BaseOperator

/**
  * Created by okay on 2017/4/10.
  *
  * 该算子将N*M的原始矩阵转换为N*K的矩阵，N为训练数据个数，M为原始数据维度，K为cluster数量
  */
class RBFBuilder extends TransformBase {
  this.operatorName = "rbf"

  //依次为：isOn,k@KMeans(聚类的cluster个数对于特征数量的占比),maxIter@KMeans,径向函数半径(高斯函数标准差)与每个cluster最边缘点距该cluster中心距离的比率
  this.params = Array(1.0, 1E-3, 20.0, 0.2)

  this.empiricalParams = Array(1.0, 1E-3, 20.0, 0.2)
  this.paramBoundaries = Array((0.0, 1.0), (1E-4, 1.0), (1.0, 1000.0), (0.01, 10.0))
  this.empiricalParamPaces = Array(0.5, 1E-4, 1.0, 0.01)
  this.paramTypes = Array(BaseOperator.PARAM_TYPE_BOOLEAN, BaseOperator.PARAM_TYPE_DOUBLE, BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_DOUBLE)

  private var clusterCenters: Array[Array[Double]] = _
  private var radiuses: Array[Double] = _

  /**
    * 运行数据处理算子
    *
    * @param data 数据（包含X,y）
    * @return 经过处理后的数据
    */
  override def run(data: DataFrame): DataFrame = {

    transform(data)
  }

  /**
    * 对数据进行transform
    *
    * @param data 数据（包含X,y）
    * @return transform后的数据
    */
  override def transform(data: DataFrame): DataFrame = {
    val transformedData = data.rdd.map {
      row =>
        val x = (for (i <- 0 until row.length) yield row(i).asInstanceOf[Double]).toArray
        val transformedRow = for (i <- clusterCenters.indices) yield {

        }

        row
    }

    data.sparkSession.createDataFrame(transformedData, ContextHolder.buildSchema(radiuses.length))
  }

  /**
    * 输出transformer的主要属性，以便别的程序可以利用这些属性，对新数据进行transform
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  override def explain(out: BufferedWriter) {
    for (i <- clusterCenters.indices) {
      out.write(radiuses(i).toString)
      out.write("\t")

      val clusterCenter = clusterCenters(i)
      for (j <- 0 until clusterCenter.length - 1) {
        out.write(clusterCenter(j).toString)
        out.write(",")
      }
      out.write(clusterCenter.last.toString)

      out.newLine()
    }

    out.flush()
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[RBFBuilder]
}
