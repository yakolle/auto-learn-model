package org.automl.model.operators.data.transform

import java.io.{BufferedWriter, IOException}

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.{DataFrame, Row}
import org.automl.model.operators.BaseOperator


/**
  * Created by zhangyikuo on 2017/1/18.
  */
class OutlierHandler extends TransformBase {
  this.operatorName = "outlier"
  this.procedureType = "preprocess"

  this.params = (1.0 +: Array.fill(OutlierHandler.getHandlerNum)(0.0)).updated(2, 1.0)

  //outlier识别和处理方法的组合，one_hot形式
  this.empiricalParams = (1.0 +: Array.fill(OutlierHandler.getHandlerNum)(0.0)).updated(2, 1.0)
  this.paramBoundaries = Array.fill(OutlierHandler.getHandlerNum + 1)((0.0, 1.0))
  this.empiricalParamPaces = Array.fill(OutlierHandler.getHandlerNum + 1)(0.5)
  this.paramTypes = Array.fill(OutlierHandler.getHandlerNum + 1)(BaseOperator.PARAM_TYPE_BOOLEAN)

  private val relativeError = 0.01

  private var means: Array[Double] = _
  private var stds: Array[Double] = _
  //4分位数第1个分位数
  private var q1s: Array[Double] = _
  private var medians: Array[Double] = _
  //4分位数第3个分位数
  private var q3s: Array[Double] = _

  /**
    * 运行数据处理算子
    *
    * @param data 数据（包含X,y），其中X并非Vector[Double]类型，其中的每个一x为单独的一列
    * @return 经过处理后的数据
    */
  override def run(data: DataFrame): DataFrame = {
    val identifyMethod = OutlierHandler.extractHandlerParams(params.tail)._1
    if (OutlierHandler.IDENTIFY_METHOD_GAUSSIAN_MEAN == identifyMethod) {
      val statisticSummary = Statistics.colStats(data.rdd.map(rec => Vectors.dense(rec.toSeq.toArray.asInstanceOf[Array[Double]])))
      means = statisticSummary.mean.toArray.dropRight(1)
      stds = statisticSummary.variance.toArray.dropRight(1).map(math.sqrt)
    } else if (OutlierHandler.IDENTIFY_METHOD_GAUSSIAN_MEDIAN == identifyMethod) {
      stds = Statistics.colStats(data.rdd.map(rec => Vectors.dense(rec.toSeq.toArray.asInstanceOf[Array[Double]]))).variance.toArray
        .dropRight(1).map(math.sqrt)
      medians = data.columns.filter(_ != "label").map(data.stat.approxQuantile(_, Array(0.5), relativeError).head)
    } else {
      val quantiles = data.columns.filter(_ != "label").map(data.stat.approxQuantile(_, Array(0.25, 0.5, 0.75), relativeError))
      q1s = quantiles.map(_ (0))
      medians = quantiles.map(_ (1))
      q3s = quantiles.map(_ (2))
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
    val (identifyMethod, identifyParam) = OutlierHandler.extractHandlerParams(params.tail)

    val transformedData = if (OutlierHandler.IDENTIFY_METHOD_GAUSSIAN_MEAN == identifyMethod)
      data.rdd.map { line =>
        Row.fromSeq((for (i <- stds.indices) yield {
          val outlierSpan = identifyParam * stds(i)
          math.min(math.max(line(i).asInstanceOf[Double], means(i) - outlierSpan), means(i) + outlierSpan)
        }) :+ line(line.length - 1).asInstanceOf[Double])
      } else if (OutlierHandler.IDENTIFY_METHOD_GAUSSIAN_MEDIAN == identifyMethod)
      data.rdd.map { line =>
        Row.fromSeq((for (i <- stds.indices) yield {
          val outlierSpan = identifyParam * stds(i)
          math.min(math.max(line(i).asInstanceOf[Double], medians(i) - outlierSpan), medians(i) + outlierSpan)
        }) :+ line(line.length - 1).asInstanceOf[Double])
      }
    else if (OutlierHandler.IDENTIFY_METHOD_TUKEY == identifyMethod)
      data.rdd.map { line =>
        Row.fromSeq((for (i <- q1s.indices) yield {
          val outlierSpan = identifyParam * (q3s(i) - q1s(i))
          math.min(math.max(line(i).asInstanceOf[Double], q1s(i) - outlierSpan), q3s(i) + outlierSpan)
        }) :+ line(line.length - 1).asInstanceOf[Double])
      }
    else data.rdd

    data.sparkSession.createDataFrame(transformedData, data.schema)
  }

  /**
    * 输出transformer的主要属性，以便别的程序可以利用这些属性，对新数据进行transform
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  override def explain(out: BufferedWriter) {
    val (identifyMethod, identifyParam) = OutlierHandler.extractHandlerParams(params.tail)

    if (OutlierHandler.IDENTIFY_METHOD_GAUSSIAN_MEAN == identifyMethod) {
      out.write("gaussian_mean")
      out.write("\t")
      out.write(identifyParam.toString)
      out.newLine()

      for (i <- means.indices) {
        out.write(means(i).toString)
        out.write("\t")
        out.write(stds(i).toString)
        out.newLine()
      }
    } else if (OutlierHandler.IDENTIFY_METHOD_GAUSSIAN_MEDIAN == identifyMethod) {
      out.write("gaussian_median")
      out.write("\t")
      out.write(identifyParam.toString)
      out.newLine()

      for (i <- medians.indices) {
        out.write(medians(i).toString)
        out.write("\t")
        out.write(stds(i).toString)
        out.newLine()
      }
    } else {
      out.write("tukey")
      out.write("\t")
      out.write(identifyParam.toString)
      out.newLine()

      for (i <- medians.indices) {
        out.write(q1s(i).toString)
        out.write("\t")
        out.write(medians(i).toString)
        out.write("\t")
        out.write(q3s(i).toString)
        out.newLine()
      }
    }

    out.flush()
  }

  /**
    * 更新超参数，如果算子不需要参数可以不用重写该方法，否则必须重写该方法
    *
    * @param params 要更新的超参数
    */
  override def updateParam(params: Array[Double]) {
    this.params = if (0.0 == params.head) {
      this.on = false
      Array.fill(getParamNum)(0.0)
    } else {
      this.on = true
      var isChosen = false
      1.0 +: params.tail.map {
        param =>
          if (1.0 == param) {
            if (isChosen) 0.0
            else {
              isChosen = true
              1.0
            }
          } else 0.0
      }
    }
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[OutlierHandler]
}

object OutlierHandler {
  //识别的方法，用高斯分布的长尾异常值检测方法（均值、标准差）
  val IDENTIFY_METHOD_GAUSSIAN_MEAN = 1
  //识别的方法，用高斯分布的长尾异常值检测方法（中位数、标准差）
  val IDENTIFY_METHOD_GAUSSIAN_MEDIAN = 2
  //Tukey Test
  val IDENTIFY_METHOD_TUKEY = 3

  /**
    * 获取异常值处理算子的个数（每个算子实际上是替换方法和识别方法及参数的组合，one_hot编码）
    *
    * @return 异常值处理算子的个数
    */
  def getHandlerNum: Int = 4 + 4 + 7

  /**
    * 从params中提取异常值替换方法，识别方法及相应参数
    *
    * @param params 参数列表，实际上是替换方法和识别方法及参数的组合，one_hot编码
    * @return 返回值格式为(异常值替换方法,异常值识别方法,异常值识别方法对应参数)
    */
  def extractHandlerParams(params: Array[Double]): (Int, Double) = {
    val index = params.indexWhere(_ == 1.0)

    index match {
      case 0 => (IDENTIFY_METHOD_GAUSSIAN_MEAN, 2.0)
      case 1 => (IDENTIFY_METHOD_GAUSSIAN_MEAN, 2.5)
      case 2 => (IDENTIFY_METHOD_GAUSSIAN_MEAN, 3.0)
      case 3 => (IDENTIFY_METHOD_GAUSSIAN_MEAN, 3.5)

      case 4 => (IDENTIFY_METHOD_GAUSSIAN_MEDIAN, 2.0)
      case 5 => (IDENTIFY_METHOD_GAUSSIAN_MEDIAN, 2.5)
      case 6 => (IDENTIFY_METHOD_GAUSSIAN_MEDIAN, 3.0)
      case 7 => (IDENTIFY_METHOD_GAUSSIAN_MEDIAN, 3.5)

      case 8 => (IDENTIFY_METHOD_TUKEY, 1.0)
      case 9 => (IDENTIFY_METHOD_TUKEY, 1.5)
      case 10 => (IDENTIFY_METHOD_TUKEY, 2.0)
      case 11 => (IDENTIFY_METHOD_TUKEY, 2.5)
      case 12 => (IDENTIFY_METHOD_TUKEY, 3.0)
      case 13 => (IDENTIFY_METHOD_TUKEY, 3.5)
      case 14 => (IDENTIFY_METHOD_TUKEY, 4.0)
    }
  }
}