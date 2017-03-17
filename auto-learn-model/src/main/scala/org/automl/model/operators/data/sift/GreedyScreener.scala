package org.automl.model.operators.data.sift

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.DataFrame
import org.automl.model.operators.BaseOperator
import org.automl.model.operators.model.validation.AUCValidation
import org.automl.model.utils.DataTransformUtil

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

/**
  * Created by zhangyikuo on 2017/1/12.
  *
  * 该算子是一个backward greedy策略的筛选算子，需要配合批量forward类型的选择算子（比如lasso）共同作用，起到shrink的作用
  */
class GreedyScreener extends SiftFeaturesBase {
  this.operatorName = "greedyScreen"

  private var trainer = new LogisticRegression().setMaxIter(100).setRegParam(0.0).setElasticNetParam(0.0)

  //增益幅度阈值，如果增益低于该阈值，停止screen
  private val gainThreshold = 1E-3
  //gainRatio的增益遗忘因子，差不多会将3步及之前的10倍于gainThreshold的增益降到gainThreshold以下
  private val forgottenFactor = 1 - math.pow(0.1, 1.0 / 3)
  //greedy beam search的beam size
  private val beamSize = 3

  //动态增益比率，按照随机逼近（遗忘算法）学习过程进行更新
  private var gainRatio = gainThreshold

  //选择后的特征集合
  private var featureIDs: Array[String] = _

  private def getNextGeneration(data: DataFrame, parents: Array[(Array[String], Double)],
                                maxPerChildNum: Int): Array[(Array[String], Double)] = {
    val children = new mutable.HashMap[Array[String], Double]

    for (i <- parents.indices) {
      val (pFeatureNames, parentAUC) = parents(i)
      val pAUC = if (parentAUC <= 0) {
        val pTrainData = DataTransformUtil.selectFeaturesFromAssembledData(data, pFeatureNames)
        AUCValidation.calcAUC(pTrainData, trainer.fit(pTrainData))
      } else parentAUC
      parents(i) = (pFeatureNames, pAUC)

      val droppingColBuffer = new ArrayBuffer[(String, Double)]
      pFeatureNames.foreach {
        feature =>
          val cData = DataTransformUtil.selectFeaturesFromAssembledData(data, pFeatureNames.filter(_ != feature))
          val cAUC = AUCValidation.calcAUC(cData, trainer.fit(cData))
          if (cAUC > pAUC + gainThreshold) droppingColBuffer += ((feature, cAUC))
      }
      children ++= droppingColBuffer.sortBy(_._2)(Ordering[Double].reverse).take(maxPerChildNum).map {
        case (droppingColName, curAUC) =>
          (pFeatureNames.filter(_ != droppingColName), curAUC)
      }
    }

    children.toArray
  }


  /**
    * 运行特征筛选算子，用LR模型直接评估，评估指标用AUC
    *
    * @param data 数据（包含X,y），其中X为Vector[Double]类型
    * @return 筛选后的数据及特征，返回值为筛选后的数据
    */
  override def run(data: DataFrame): DataFrame = {
    gainRatio = math.max(gainThreshold, gainRatio)
    val maxSearchDepth = math.max(Random.nextInt(math.ceil(gainRatio / gainThreshold).toInt), 1)

    var totalGain = 0.0
    var depth = 0

    var parents = Array((DataTransformUtil.extractFeatureNamesFromAssembledData(data), -1.0))
    var children = getNextGeneration(data, parents, beamSize)
    while (children.nonEmpty && depth <= maxSearchDepth) {
      children = children.sortBy(_._2)(Ordering[Double].reverse)
      children = if (children.length > beamSize) children.take(beamSize) else children
      totalGain += children.head._2 - parents.head._2
      depth += 1

      parents = children
      if (depth <= maxSearchDepth)
        children = getNextGeneration(data, children, math.ceil(children.length.toDouble / beamSize.toDouble).toInt)
    }

    if (depth > 0) gainRatio = (1 - forgottenFactor) * gainRatio + forgottenFactor * totalGain / depth.toDouble

    this.featureIDs = parents.head._1
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
  override def clone: BaseOperator = {
    val copy = super.clone.asInstanceOf[GreedyScreener]
    copy.trainer = new LogisticRegression().setMaxIter(100).setRegParam(0.0).setElasticNetParam(0.0)
    copy
  }
}
