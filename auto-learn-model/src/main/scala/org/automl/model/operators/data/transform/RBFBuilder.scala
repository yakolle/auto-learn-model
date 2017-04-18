package org.automl.model.operators.data.transform

import java.io.{BufferedWriter, IOException}

import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}
import org.automl.model.context.ContextHolder
import org.automl.model.operators.BaseOperator

/**
  * Created by okay on 2017/4/10.
  *
  * 该算子将N*M的原始矩阵转换为N*K的矩阵，N为训练数据个数，M为原始数据维度，K为cluster数量
  */
class RBFBuilder extends TransformBase {
  this.operatorName = "rbf"

  //依次为：isOn,k@KMeans,maxIter@KMeans,径向函数半径(高斯函数标准差)与每个cluster最边缘点距该cluster中心距离的比率
  this.params = Array(1.0, 10.0, 20.0, 0.2)

  this.empiricalParams = Array(1.0, 10.0, 20.0, 0.2)
  this.paramBoundaries = Array((0.0, 1.0), (2.0, 1E5), (1.0, 1000.0), (0.01, 10.0))
  this.empiricalParamPaces = Array(0.5, 1.0, 1.0, 0.01)
  this.paramTypes = Array(BaseOperator.PARAM_TYPE_BOOLEAN, BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_INT, BaseOperator.PARAM_TYPE_DOUBLE)

  private var dataSize = Long.MaxValue
  private var model: KMeansModel = _
  private var radiuses: Array[Double] = _

  /**
    * 运行数据处理算子
    *
    * @param data 数据（包含X,y），其中X为Vector[Double]类型
    * @return 经过处理后的数据
    */
  override def run(data: DataFrame): DataFrame = {
    if (Long.MaxValue == dataSize) {
      dataSize = data.count()
      paramBoundaries(1) = (2.0, dataSize.toDouble)
    }

    model = new KMeans().setK(params(1).toInt).setMaxIter(params(2).toInt).fit(data)

    val centers = model.clusterCenters
    val distUDF = udf((features: Vector, clusterIndex: Int) => Vectors.sqdist(features, centers(clusterIndex)))
    val rs = model.transform(data).withColumn("dist", distUDF(col("features"), col("prediction")))
      .groupBy("prediction").agg(max("dist")).collect().sortBy(_.getAs[Int](0)).map(_.getAs[Double](1) * params.last)

    val minRadius = rs.minBy(radius => if (radius <= 0.0) Double.MaxValue else radius)
    radiuses = rs.map(radius => if (radius <= 0.0) minRadius else radius)
    transform(data)
  }

  /**
    * 对数据进行transform
    *
    * @param data 数据（包含X,y），其中X为Vector[Double]类型
    * @return transform后的数据
    */
  override def transform(data: DataFrame): DataFrame = {
    val clusterCenters = model.clusterCenters
    val rs = radiuses

    val transformedData = data.rdd.map { row =>
      val feature = row.getAs[Vector](0)
      Row(Vectors.dense((for (i <- clusterCenters.indices) yield {
        val dist = Vectors.sqdist(feature, clusterCenters(i))
        math.exp(-dist * dist / (2 * rs(i) * rs(i)))
      }).toArray), if (row(1).isInstanceOf[Int]) row.getInt(1).toDouble else row.getDouble(1))
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
    val clusterCenters = model.clusterCenters
    for (i <- clusterCenters.indices) {
      out.write(radiuses(i).toString)
      out.write("\t")

      val clusterCenter = clusterCenters(i)
      val end = clusterCenter.size - 1
      for (j <- 0 until end) {
        out.write(clusterCenter(j).toString)
        out.write(",")
      }
      out.write(clusterCenter(end).toString)

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
