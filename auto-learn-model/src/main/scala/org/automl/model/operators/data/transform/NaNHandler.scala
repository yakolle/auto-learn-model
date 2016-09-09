package org.automl.model.operators.data.transform

import java.io.{BufferedWriter, IOException}

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.utils.DataStatisticUtil

/**
  * Created by zhangyikuo on 2017/1/17.
  */
class NaNHandler extends TransformBase {
  this.operatorName = "NaN"
  this.procedureType = "preprocess"

  this.on = true

  private var meanMap: Map[String, Double] = _
  private var medianMap: Map[String, Double] = _

  //只有两个值0.0或1.0，表示用mean或median来填充缺失值
  private var params: Array[Double] = _

  private val empiricalParams: Array[Double] = Array(0.0)

  /**
    * 运行数据处理算子
    *
    * @param data 数据（包含X,y），其中X并非Vector[Double]类型，其中的每个一x为单独的一列
    * @return 经过处理后的数据
    */
  override def run(data: DataFrame): DataFrame = {
    if (0.0 == params.head) this.meanMap = DataStatisticUtil.calcStatistic(data, "mean")
    else this.medianMap = DataStatisticUtil.calcMedian(data)

    transform(data)
  }

  /**
    * 对数据进行transform
    *
    * @param data 数据（包含X,y），其中X并非Vector[Double]类型，其中的每个一x为单独的一列
    * @return transform后的数据
    */
  override def transform(data: DataFrame): DataFrame = if (0.0 == params.head) data.na.fill(meanMap) else data.na.fill(medianMap)

  /**
    * 输出transformer的主要属性，以便别的程序可以利用这些属性，对新数据进行transform
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  override def explain(out: BufferedWriter) {
    out.write(if (0.0 == params.head) "mean" else "median")
    out.flush()
  }

  /**
    * 获取超参数个数
    *
    * @return 超参数个数，如果没有超参数，返回0
    */
  override def getParamNum: Int = empiricalParams.length

  /**
    * 获取超参数当前值
    *
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的当前值
    */
  override def getCurrentParam(paramIndex: Int): Double = params(paramIndex)

  /**
    * 获取warm start点（超参数经验搜索起始点）
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验值（warm start点）
    */
  override def getEmpiricalParam(data: DataFrame, paramIndex: Int): Double = empiricalParams(paramIndex)

  /**
    * 更新超参数，如果算子不需要参数可以不用重写该方法，否则必须重写该方法
    *
    * @param params 要更新的超参数
    */
  override def updateParam(params: Array[Double]) {
    this.params = params
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[NaNHandler]
}
