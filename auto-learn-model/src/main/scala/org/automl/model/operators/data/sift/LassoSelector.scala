package org.automl.model.operators.data.sift

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.utils.DataTransformUtil

/**
  * Created by zhangyikuo on 2016/10/12.
  */
class LassoSelector extends SiftFeaturesBase {
  this.operatorName = "lasso"

  //just lamda
  this.params = Array(0.0)

  //warm start，lamda经验搜索起始点
  this.empiricalParams = Array(1E-3)
  //lamda的搜索范围
  this.paramBoundaries = Array((0.0, 1.0))
  //lamda的经验搜索步幅
  this.empiricalParamPaces = Array(1E-6)
  this.paramTypes = Array(BaseOperator.PARAM_TYPE_DOUBLE)

  private val maxIterations = 100
  private val zeroDomain = 1E-6

  //选择后的特征集合
  private var featureIDs: Array[String] = _

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

    this.featureIDs = coefficients match {
      case vector: SparseVector if 0 == vector.indices.length => colNames.take(1)
      case _ => if (selectedColNames.length <= 0) Array(colNames(coefficients.argmax)) else selectedColNames.toArray
    }
    transform(data)
  }

  /**
    * 获取该算子目前所选择的特征ID数组
    *
    * @return 算子目前所选择的特征ID数组
    */
  override def getFeatureIDs: Array[String] = this.featureIDs

  /**
    * 重载Object的clone方法，子类如果有参数或者一些引用型(AnyRef)的属性，必须重写该方法
    *
    * @return 复制后的对象
    */
  override def clone: BaseOperator = super.clone.asInstanceOf[LassoSelector]
}
