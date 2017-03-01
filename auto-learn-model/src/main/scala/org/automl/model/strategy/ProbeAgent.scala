package org.automl.model.strategy

import org.automl.model.context.ContextHolder
import org.automl.model.operators.data.bagging.BaggingBase
import org.automl.model.operators.data.balance.BalanceBase
import org.automl.model.operators.data.evaluation.EvaluationBase
import org.automl.model.operators.data.format.FormatBase
import org.automl.model.operators.data.sift.SiftFeaturesBase
import org.automl.model.operators.data.transform.TransformBase
import org.automl.model.operators.model.train.TrainBase
import org.automl.model.operators.model.validation.ValidationBase
import org.automl.model.strategy.scheduler.ProbeSchedulerBase

/**
  * Created by zhangyikuo on 2016/8/19.
  */
class ProbeAgent extends Runnable {
  var task: ProbeTask = _
  var scheduler: ProbeSchedulerBase = _

  //控制agent是否停止探索
  private var stopFlag = false

  def terminate() {
    stopFlag = true
  }

  override def run() {
    while (!stopFlag) {
      run(task)
      scheduler.onlineLearn(task.getParams :+ task.calcFinalValidation)
      ContextHolder.feedback(task)
      task = scheduler.getNextProbeTask(task, ContextHolder.getParams)
    }
  }

  def run(task: ProbeTask) = {
    val operatorChain = task.getOperatorChain
    val operatorNum = operatorChain.length

    //任务的resurge操作
    val savepoint = task.getSavepoint
    if (savepoint.isEmpty) task.runPoint = 0
    else {
      var runPointIndex = savepoint.indexWhere(_._1 > task.runPoint)
      if (0 == runPointIndex) task.runPoint = 0
      else {
        runPointIndex = if (runPointIndex < 0) savepoint.length - 1 else runPointIndex - 1
        savepoint.trimEnd(savepoint.length - runPointIndex - 1)
        val savepointTuple = savepoint(runPointIndex)
        //任务从savepoint点后开始运行
        task.runPoint = savepointTuple._1 + 1
        task.trainData = savepointTuple._2
        task.testData = savepointTuple._3
      }
    }
    if (0 == task.runPoint) {
      task.trainData = null
      task.testData = null
      task.getSavepoint.clear
    }

    //按照算子序列的顺序，从task.runPoint点开始顺序运行算子
    while (task.runPoint < operatorNum) {
      val i = task.runPoint
      val operator = operatorChain(i)
      operator match {
        case operator: EvaluationBase => operator.run(task.trainData)
        case operator: BaggingBase =>
          val res = operator.run(task.data)
          task.trainData = res._1
          task.testData = res._2
        case operator: BalanceBase => task.trainData = operator.run(task.trainData)
        case operator: FormatBase =>
          task.trainData = operator.run(task.trainData)
          task.testData = operator.run(task.testData)
        case operator: TransformBase =>
          if (operator.isOn) {
            task.trainData = operator.run(task.trainData).cache()
            task.testData = operator.transform(task.testData).cache()
          }
          //transform结束后，对数据进行保存
          savepoint += ((i, task.trainData, task.testData))
        case operator: SiftFeaturesBase =>
          task.trainData = operator.run(task.trainData).cache()
          task.testData = operator.transform(task.testData).cache()
          //特征筛选结束后，对数据进行保存
          savepoint += ((i, task.trainData, task.testData))
        case operator: TrainBase => operator.run(task.trainData)
        case operator: ValidationBase => operator.run(task.trainData, task.getTrainer.getModel, task.testData)
      }
      task.runPoint += 1
    }
  }
}
