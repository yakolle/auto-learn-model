package org.automl.model.operators.data.sift

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.utils.DataTransformUtil

/**
  * Created by zhangyikuo on 2016/10/12.
  */
class LassoSelector extends SiftFeaturesBase {
  this.operatorName = "lasso"

  private val maxIterations = 100
  private val zeroDomain = 1E-6

  //just lamda
  private var params = Array(0.0)
  //选择后的特征集合
  private var featureIDs: Array[String] = _

  //warm start，lamda经验搜索起始点
  private val empiricalParams = Array(1E-3)
  //lamda的搜索范围
  private val paramBoundaries = Array((0.0, 1.0))
  //lamda的经验搜索步幅
  private val empiricalParamPaces = Array(1E-6)

  /**
    * 运行特征筛选算子
    *
    * @param data 数据（包含X,y），其中X为Vector[Double]类型
    * @return 筛选后的数据及特征，返回值为筛选后的数据
    */
  override def run(data: DataFrame): DataFrame = {
    val model = new LogisticRegression().setMaxIter(maxIterations).setRegParam(params(0)).setElasticNetParam(1.0)
      .setFitIntercept(false).fit(data)

    val coefficients = model.coefficients
    val colNames = DataTransformUtil.extractFeatureNamesFromAssembledData(data)
    val selectedColNames = for (i <- 0 until coefficients.size; if math.abs(coefficients(i)) > zeroDomain)
      yield colNames(i)

    this.featureIDs = selectedColNames.toArray
    transform(data)
  }

  /**
    * 对数据进行特征筛选
    *
    * @param data 数据（包含X,y），其中X为Vector[Double]类型
    * @return 进过特征筛选后的数据
    */
  override def transform(data: DataFrame): DataFrame = DataTransformUtil.selectFeaturesFromAssembledData(data, this.featureIDs)

  /**
    * 获取该算子目前所选择的特征ID数组
    *
    * @return 算子目前所选择的特征ID数组
    */
  override def getFeatureIDs: Array[String] = this.featureIDs

  /**
    * 获取超参数个数
    *
    * @return 超参数个数，如果没有超参数，返回0
    */
  override def getParamNum: Int = empiricalParams.length

  /**
    * 获取超参数的搜索范围
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的搜索范围，返回值为(minParam,maxParam)
    */
  override def getParamBoundary(data: DataFrame, paramIndex: Int): (Double, Double) = paramBoundaries(paramIndex)

  /**
    * 获取超参数的经验搜索步幅
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验搜索步幅
    */
  override def getEmpiricalParamPace(data: DataFrame, paramIndex: Int): Double = empiricalParamPaces(paramIndex)

  /**
    * 获取warm start点（超参数经验搜索起始点）
    *
    * @param data       数据（包含X,y）
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的经验值（warm start点）
    */
  override def getEmpiricalParam(data: DataFrame, paramIndex: Int): Double = empiricalParams(paramIndex)

  /**
    * 获取超参数当前值
    *
    * @param paramIndex 第几个（从0开始）超参数
    * @return 第paramIndex个超参数的当前值
    */
  override def getCurrentParam(paramIndex: Int): Double = params(paramIndex)

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
  override def clone: BaseOperator = super.clone.asInstanceOf[LassoSelector]
}
