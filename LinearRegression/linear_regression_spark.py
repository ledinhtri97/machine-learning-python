"""LinearRegression"""
from __future__ import division, print_function, unicode_literals
from pyspark import SparkContext, SparkConf

"""METHOD ONE"""
# def myFunc(s):
#     words = s.split(" ")
#     return len(words)
#
#
# sc = SparkContext(...)
# sc.textFile("file.txt").map(myFunc)

"""METHOD TWO"""
# class MyClass(object):
#     def func(self, s):
#         return s
#     def doStuff(self, rdd):
#         return rdd.map(self.func)

if __name__ == "__main__":
    APP_NAME = "PYTHON_SPARK"
    MASTER = "local"

    # Initializing Spark
    conf = SparkConf().setAppName(APP_NAME).setMaster(MASTER)
    sc = SparkContext(conf= conf)

    # Parallelized Collections
    data = [1, 2, 1, 5, 4]
    distData = sc.parallelize(data)

    # External Datasets
    distFile = sc.textFile("D://input.txt")

    # Basics
    lineLengths = distFile.map(lambda s: len(s))
    totalLength = lineLengths.reduce(lambda a,b : a+b)

    lineLengths.persist()

    # Working with Key-Value Pairs
    pairs = lineLengths.map(lambda s: (s, 1))
    counts = pairs.reduceByKey(lambda a, b: a+b)

    # Test
    print(totalLength)
