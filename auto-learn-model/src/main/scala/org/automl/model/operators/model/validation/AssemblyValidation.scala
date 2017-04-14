package org.automl.model.operators.model.validation

/**
  * Created by zhangyikuo on 2017/1/10.
  *
  * 这是一个工具对象，用于融合各个常见的验证值，计算得出最终的验证值
  */
object AssemblyValidation {
  //验证项权重的5元组列表，如果训练集验证值为x，测试集验证值为y，其格式为(x^2,y^2,x*y,x,y)
  private var vWeights: Array[(Double, Double, Double, Double, Double)] = _
  //训练过程中验证项权重的5元组列表
  private var trainVWeights: Array[(Double, Double, Double, Double, Double)] = _
  //训练过程中验证值和训练结束后验证值权重5元组
  private var trainingAndTrainedVWeight: (Double, Double, Double, Double, Double) = _

  /**
    * 设置验证项权重5元组列表，该权重列表一般由外部环境设置
    *
    * @param vWeights 验证项权重5元组列表，如果训练集验证值为x，测试集验证值为y，其格式为(x^2,y^2,x*y,x,y)
    */
  def setValidatorWeights(vWeights: Array[(Double, Double, Double, Double, Double)]) {
    this.vWeights = vWeights
  }

  /**
    * 设置训练过程中验证项权重5元组列表，该权重列表一般由外部环境设置
    *
    * @param trainVWeights 验证项权重5元组列表，如果训练集验证值为x，测试集验证值为y，其格式为(x^2,y^2,x*y,x,y)
    */
  def setTrainValidatorWeights(trainVWeights: Array[(Double, Double, Double, Double, Double)]) {
    this.trainVWeights = trainVWeights
  }

  /**
    * 设置训练过程中验证值和训练结束后验证值权重5元组，该权重列表一般由外部环境设置
    *
    * @param trainingAndTrainedVWeight 训练过程中验证值和训练结束后验证值权重5元组，如果训练过程中验证值为x，
    *                                  训练结束后验证值为y，其格式为(x^2,y^2,x*y,x,y)
    */
  def setTrainingAndTrainedValidationWeight(trainingAndTrainedVWeight: (Double, Double, Double, Double, Double)) {
    this.trainingAndTrainedVWeight = trainingAndTrainedVWeight
  }

  /**
    * 根据融合权重，计算融合验证值
    *
    * @param validations 验证值列表，列表里的元素格式为(trainValidation,testValidation)
    * @return 融合验证值
    */
  def assembleValidation(validations: Array[(Double, Double)]): Double = assembleValidation(vWeights, validations)

  /**
    * 根据融合权重，计算训练过程中融合验证值
    *
    * @param validations 验证值列表，列表里的元素格式为(trainValidation,testValidation)
    * @return 训练过程中融合验证值
    */
  def assembleTrainValidation(validations: Array[(Double, Double)]): Double = assembleValidation(trainVWeights, validations)

  /**
    * 根据融合权重，计算融合验证值
    *
    * @param vWeights    融合权重
    * @param validations 验证值列表，列表里的元素格式为(trainValidation,testValidation)
    * @return 融合验证值
    */
  private def assembleValidation(vWeights: Array[(Double, Double, Double, Double, Double)], validations: Array[(Double, Double)]) = {
    var assembledValidation = 0.0
    for (i <- vWeights.indices) {
      val vWeight = vWeights(i)
      val vTuple = validations(i)
      assembledValidation += (vTuple._1 * (vWeight._1 * vTuple._1 + vWeight._4 + vWeight._3 * vTuple._2)
        + vTuple._2 * (vWeight._2 * vTuple._2 + vWeight._5))
    }
    assembledValidation
  }

  /**
    * 根据融合权重，融合训练过程中验证值和训练结束后验证值
    *
    * @param trainingV 训练过程中验证值
    * @param trainedV  训练结束后验证值
    * @return 融合权重
    */
  def assembleTrainingAndTrainedValidation(trainingV: Double, trainedV: Double): Double =
    (trainingV * (trainingAndTrainedVWeight._1 * trainingV + trainingAndTrainedVWeight._4 + trainingAndTrainedVWeight._3 * trainedV)
      + trainedV * (trainingAndTrainedVWeight._2 * trainedV + trainingAndTrainedVWeight._5))

}
