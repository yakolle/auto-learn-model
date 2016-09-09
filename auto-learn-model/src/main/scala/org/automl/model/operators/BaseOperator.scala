package org.automl.model.operators

import org.apache.spark.sql.DataFrame

/**
  * Created by zhangyikuo on 2016/8/18.
  * 不要随意修改BaseOperator成员函数的默认行为，以免对子类造成影响
  */
abstract class BaseOperator extends Cloneable {
  //算子名称
  var operatorName: String = "base"
  //算子类型
  var operatorType: String = "base"
  //过程类型
  var procedureType: String = "base"

  /**
    * 获取算子的全称
    *
    * @return 算子的全称
    */
  def getCanonicalName: String = procedureType + "." + operatorType + "." + operatorName

  /**
    * 获取超参数个数
    *
    * @return 超参数个数，如果没有超参数，返回0
    */
  def getParamNum: Int = 0

  /**
    * 获取超参数的搜索范围
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的搜索范围，返回值为(minParam,maxParam)
    */
  def getParamBoundary(data: DataFrame, paramIndex: Int): (Double, Double) = (0.0, 1.0)

  /**
    * 获取超参数的类型，只有三种，double、int、boolean，默认为double
    *
    * @param paramIndex 第几个（从0开始）超参数
    * @return 超参数的类型
    */
  def getParamType(paramIndex: Int): Int = BaseOperator.PARAM_TYPE_DOUBLE

  /**
    * 获取超参数的经验搜索步幅
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验搜索步幅
    */
  def getEmpiricalParamPace(data: DataFrame, paramIndex: Int): Double = 1E-6

  /**
    * 获取warm start点（超参数经验搜索起始点）
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验值（warm start点）
    */
  def getEmpiricalParam(data: DataFrame, paramIndex: Int): Double = 1E-3

  /**
    * 获取超参数当前值
    *
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的当前值
    */
  def getCurrentParam(paramIndex: Int): Double = 0.0

  /**
    * 更新超参数，如果算子不需要参数可以不用重写该方法，否则必须重写该方法
    *
    * @param params 要更新的超参数
    */
  def updateParam(params: Array[Double]) {}

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[BaseOperator]
}

object BaseOperator {
  val PARAM_TYPE_DOUBLE = 1
  val PARAM_TYPE_INT = 2
  val PARAM_TYPE_BOOLEAN = 3
}

