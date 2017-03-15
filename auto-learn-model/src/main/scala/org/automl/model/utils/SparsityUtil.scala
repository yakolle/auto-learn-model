package org.automl.model.utils

/**
  * Created by zhangyikuo on 2017/3/15.
  */
object SparsityUtil {
  /**
    * 找到间隔最大两个数的索引
    *
    * @param data   原数据
    * @param curMin 最小值
    * @param curMax 最大值
    * @return 返回值格式为:(是否均衡,间隔最大两个数的索引1,间隔最大两个数的索引2)
    */
  def findMaxGap(data: Array[Double], curMin: Double, curMax: Double) = {
    val dataLen = data.length
    //构建个dataLen桶，将data装入相应桶中
    val buckets: Array[(Int, Int)] = Array.fill(dataLen)(null)
    for (i <- data.indices) {
      val ele = data(i)
      val bucketIndex = math.min(((ele - curMin) * dataLen / (curMax - curMin)).toInt, dataLen - 1)
      val bucketTuple = buckets(bucketIndex)
      buckets(bucketIndex) = if (null == bucketTuple) (i, i)
      else if (ele > data(bucketTuple._2)) (bucketTuple._1, i)
      else if (ele < data(bucketTuple._1)) (i, bucketTuple._2)
      else bucketTuple
    }

    //找到连续空桶数最多的section
    var maxEmptyBucketNum = 0
    var start = 0
    var end = 0
    var curEmptyBucketNum = 0
    var curEmptyBucketStart = 0
    for (i <- buckets.indices) {
      if (null == buckets(i)) curEmptyBucketNum += 1
      else {
        if (curEmptyBucketNum >= maxEmptyBucketNum) {
          maxEmptyBucketNum = curEmptyBucketNum
          start = curEmptyBucketStart
          end = i
        }
        curEmptyBucketNum = 0
        curEmptyBucketStart = i
      }
    }

    //如果没有空桶，实际上buckets已是排好序的plowParam了
    if (end - start <= 1) {
      var maxSpan = Double.MinValue
      for (i <- 1 until buckets.length) {
        val curSpan = data(buckets(i)._1) - data(buckets(i - 1)._1)
        if (curSpan > maxSpan) {
          end = i
          maxSpan = curSpan
        }
      }
      (true, buckets(end - 1)._1, buckets(end)._1)
    } else (false, buckets(start)._2, buckets(end)._1)
  }
}
