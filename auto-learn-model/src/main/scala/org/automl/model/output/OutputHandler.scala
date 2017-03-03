package org.automl.model.output

import java.io.{BufferedWriter, File, FileWriter, IOException}

import org.apache.commons.io.FileUtils
import org.automl.model.operators.BaseOperator
import org.automl.model.operators.data.evaluation.EvaluationBase
import org.automl.model.operators.data.sift.SiftFeaturesBase
import org.automl.model.operators.data.transform.TransformBase
import org.automl.model.operators.model.train.TrainBase
import org.automl.model.operators.model.validation.ValidationBase

import scala.collection.mutable.ArrayBuffer

/**
  * Created by zhangyikuo on 2017/3/3.
  */
object OutputHandler {
  /**
    * 输出收敛记录
    *
    * @param convergeRecBuffer 收敛记录，参见ContextHolder.convergeRecBuffer
    * @param outputFilePath    收敛记录文件路径
    */
  def outputConvergenceRecord(convergeRecBuffer: ArrayBuffer[(Int, Double, Double, Array[Array[Double]])], outputFilePath: String) {
    val lines = new java.util.ArrayList[String]
    val strBuffer = StringBuilder.newBuilder
    for (learnRec <- convergeRecBuffer) {
      strBuffer.clear()
      strBuffer.append(learnRec._1).append("\t").append(learnRec._2).append("\t").append(learnRec._3).append("\t")
      for (bestParams <- learnRec._4) {
        for (bestParamEle <- bestParams) strBuffer.append(bestParamEle).append(",")

        strBuffer.setLength(strBuffer.length - 1)
        strBuffer.append("\t")
      }

      lines.add(strBuffer.substring(0, strBuffer.length - 1))
    }

    try
      FileUtils.writeLines(new File(outputFilePath), lines)
    catch {
      case e: IOException =>
        e.printStackTrace()
    }
  }

  /**
    * 输出搜索到的最好结果
    *
    * @param bestOperatorSequences 目前为止效果最好的搜索任务，参见ContextHolder.bestOperatorSequences
    * @param outputFilePath        搜索结果文件路径
    */
  def outputBestSearchResults(bestOperatorSequences: Array[(Array[BaseOperator], Double)], outputFilePath: String) {
    val writer = new BufferedWriter(new FileWriter(outputFilePath))
    val strBuffer = StringBuilder.newBuilder

    try {
      bestOperatorSequences.foreach {
        case (operatorChain, validation) =>
          if (null != operatorChain) {
            operatorChain.foreach {
              case operator: EvaluationBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()

                strBuffer.clear()
                operator.getEvaluations.foreach(strBuffer.append(_).append("\t"))
                writer.write(strBuffer.substring(0, strBuffer.length - 1))
                writer.newLine()

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case operator: TransformBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()
                writer.write(if (operator.isOn) "on" else "off")
                writer.newLine()

                if (operator.isOn) {
                  operator.explain(writer)
                  writer.newLine()
                }

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case operator: SiftFeaturesBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()

                strBuffer.clear()
                operator.getFeatureIDs.foreach(strBuffer.append(_).append("\t"))
                writer.write(strBuffer.substring(0, strBuffer.length - 1))
                writer.newLine()

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case operator: TrainBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()

                operator.explainModel(writer)
                writer.newLine()

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case operator: ValidationBase =>
                writer.write(operator.getCanonicalName)
                writer.newLine()

                strBuffer.clear()
                operator.getValidations.foreach {
                  case (trainValidation, testValidation) =>
                    strBuffer.append(trainValidation).append(",").append(testValidation).append("\t")
                }
                writer.write(strBuffer.substring(0, strBuffer.length - 1))
                writer.newLine()

                writer.write("-----------------------------------------------------------")
                writer.newLine()
              case _ =>
            }

            writer.write("finalValidation=" + validation)
            writer.newLine()
          }
          writer.write("===========================================================")
          writer.newLine()
          writer.newLine()
      }

      writer.close()
    } catch {
      case e: IOException =>
        e.printStackTrace()
    }
  }
}
