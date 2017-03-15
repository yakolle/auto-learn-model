package org.automl.model

import java.util.concurrent.{Executors, TimeUnit}

import org.automl.model.context.{ContextHolder, ParamHoldler, TaskBuilder}
import org.automl.model.output.OutputHandler

/**
  * Created by zhangyikuo on 2016/8/26.
  */
object MasterConsole {
  def main(args: Array[String]) {
    //初始化
    val sparkSession = TaskBuilder.initContext(args)
    ContextHolder.setSparkSession(sparkSession)

    val data = TaskBuilder.loadData(sparkSession, args)
    val operators = TaskBuilder.loadOperators(args)
    TaskBuilder.initAssemblyValidation(operators)
    TaskBuilder.initIdealValidation(operators)
    ParamHoldler.initBestOperatorSequences(TaskBuilder.bestResultNum)
    val beamSearchNum = TaskBuilder.getBeamSearchNum(sparkSession)
    val tasks = TaskBuilder.buildProbeTask(operators, data, beamSearchNum)
    val agents = TaskBuilder.buildProbeAgent(beamSearchNum)
    val scheduler = TaskBuilder.getScheduler
    for (i <- 0 until beamSearchNum) {
      agents(i).task = tasks(i)
      agents(i).scheduler = scheduler
    }

    //并行化探索
    val exeService = Executors.newFixedThreadPool(beamSearchNum)
    agents.foreach(exeService.execute(_))

    //动态学习搜索过程，自适应调整参数，判断任务收敛状态
    var currentRunTimes = 0
    while (currentRunTimes <= TaskBuilder.minIterations || (currentRunTimes <= TaskBuilder.maxIterations && !ContextHolder.hasConverged)) {
      //一般情况下，至少保证每个搜索任务（共beamSearchNum个）都探测一遍后，才开始进行学习与反馈调整
      if (currentRunTimes >= beamSearchNum) {
        scheduler.learn(ContextHolder.toDF(ParamHoldler.getParams))
        ContextHolder.adjustMaxEstimateAcceptRatio()
      }
      Thread.sleep(TaskBuilder.learnInterval)
      currentRunTimes = ParamHoldler.getRunTimes
    }

    //探索任务结束
    agents.foreach(_.terminate())
    exeService.shutdown()
    exeService.awaitTermination(beamSearchNum * TaskBuilder.learnInterval, TimeUnit.MILLISECONDS)

    OutputHandler.outputConvergenceRecord(ContextHolder.getConvergeRecords, TaskBuilder.getConvergenceRecordOutputPath)
    OutputHandler.outputBestSearchResults(ParamHoldler.getBestOperatorSequences, TaskBuilder.getBestResultsOutputPath)
  }
}