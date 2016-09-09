package org.automl.model.operators.data.transform

import java.io.{BufferedWriter, IOException}

import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator

/**
  * Created by zhangyikuo on 2016/8/19.
  * 不要随意修改TransformBase成员函数的默认行为，以免对子类造成影响
  */
abstract class TransformBase extends BaseOperator {
  this.operatorName = "transform"
  this.operatorType = "transform"
  this.procedureType = "transform"

  //该transform是否开启
  protected var on = false

  /**
    * 该transform是否开启
    *
    * @return 该transform是否开启
    */
  def isOn = on

  /**
    * 运行数据处理算子
    *
    * @param data 数据（包含X,y）
    * @return 经过处理后的数据
    */
  def run(data: DataFrame): DataFrame

  /**
    * 对数据进行transform
    *
    * @param data 数据（包含X,y）
    * @return transform后的数据
    */
  def transform(data: DataFrame): DataFrame

  /**
    * 输出transformer的主要属性，以便别的程序可以利用这些属性，对新数据进行transform
    *
    * @param out 输出流
    * @throws IOException 输出IO异常
    */
  @throws(classOf[IOException])
  def explain(out: BufferedWriter)

  /**
    * 获取超参数个数
    *
    * @return 超参数个数，如果没有超参数，返回0
    */
  override def getParamNum: Int = 1

  /**
    * 获取超参数的搜索范围
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的搜索范围，返回值为(minParam,maxParam)
    */
  override def getParamBoundary(data: DataFrame, paramIndex: Int): (Double, Double) = (0.0, 1.0)

  /**
    * 获取超参数的类型，只有三种，double、int、boolean，默认为double
    *
    * @param paramIndex 第几个（从0开始）超参数
    * @return 超参数的类型
    */
  override def getParamType(paramIndex: Int): Int = BaseOperator.PARAM_TYPE_BOOLEAN

  /**
    * 获取超参数的经验搜索步幅
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验搜索步幅
    */
  override def getEmpiricalParamPace(data: DataFrame, paramIndex: Int): Double = 0.5

  /**
    * 获取warm start点（超参数经验搜索起始点）
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验值（warm start点）
    */
  override def getEmpiricalParam(data: DataFrame, paramIndex: Int): Double = 0.0

  /**
    * 获取超参数当前值
    *
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的当前值
    */
  override def getCurrentParam(paramIndex: Int): Double = if (on) 1.0 else 0.0

  /**
    * 更新超参数，如果算子不需要参数可以不用重写该方法，否则必须重写该方法
    *
    * @param params 要更新的超参数
    */
  override def updateParam(params: Array[Double]) {
    this.on = 1.0 == params.head
  }

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[TransformBase]
}
