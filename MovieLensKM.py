import sys
import itertools
import random
from math import sqrt
from operator import add
from os.path import join, isfile, dirname

from math import sqrt

from pyspark import SparkConf, SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel

def parseRating(line):
    """
    Parses a rating record in MovieLens format timestamp userId movieId rating.
    """
    fields = line.strip().split("\t")
    return long(fields[3]) % 10, (int(fields[0]) + 10000, int(fields[1]), float(fields[2]))

def parseGenre(line):
	"""
	Parses a movie record in MovieLens format movieId|||||movieGenres .
	"""
	fields = line.strip().split("|")
	return int(fields[0]), tuple([int(x) for x in fields[5:]])
    

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
          "MovieLensALS.py movieLensDataDir"
        sys.exit(1)

  
    conf = SparkConf() \
      .setAppName("MovieLensKM") \
      .set("spark.executor.memory", "4g")
    sc = SparkContext(conf=conf)

    movieLensHomeDir = sys.argv[1]

    ratings = sc.textFile(join(movieLensHomeDir, "u.data")).map(parseRating)

    movies = sc.textFile(join(movieLensHomeDir, "u.item")).map(parseGenre)

    numRatings = ratings.count()
    numUsers = ratings.values().map(lambda r: r[0]).distinct().count()
    numMovies = ratings.values().map(lambda r: r[1]).distinct().count()
    numGenres = 19

    print "Got %d ratings from %d users on %d movies for %d genres." % (numRatings, numUsers, numMovies, numGenres)

    numPartitions = 4
    
    trainMovies = movies.filter(lambda x: x[0] % 10 < 6) \
    .values() \
    .repartition(numPartitions) \
    .cache()
        
    validateMovies = movies.filter(lambda x: x[0] % 10 >= 6 and x[0] % 10 < 8) \
    .values() \
    .repartition(numPartitions) \
    .cache()
    
    testMovies = movies.filter(lambda x: x[0] % 10 >= 8) \
    .values () \
    .cache()

    numTrainMovies = trainMovies.count()
    numValidateMovies = validateMovies.count()
    numTestMovies = testMovies.count()
     
    print "Movies data was partitioned into training: %d, validation: %d, test: %d." % (numTrainMovies, numValidateMovies, numTestMovies)
    
    bestModel = None
    bestValidationSse = float("inf")
    bestK = 0
    ks = ks = list(range(2, 20)) + list(range(20, 100, 10)) + list(range(100, 200, 25))
    
    for K in ks:
            model = KMeans.train(trainMovies, K, seed=123)
            validationSse = model.computeCost(validateMovies)
            print "SSE = %d for the model with %d clusters" % (validationSse, model.k)
            if (validationSse < bestValidationSse):
                bestModel = model
                bestValidationSse = validationSse
                bestK = model.k
                     
    testSse = bestModel.computeCost(testMovies)
    print "The best model has %d clusters, its training SSE is %d and its SSE on the test set is %d." % (bestK, validationSse, testSse)

    clusters = movies.map(lambda x: x[1]).map(model.predict)
    movieCluster = movies.map(lambda x : x[0]).zip(clusters)
            
    movieUserRatingCluster = ratings.values().map(lambda x: (x[1],(x[0], x[2]))).join(movieCluster).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1]))
        
    userClusterAvg = movieUserRatingCluster.map(lambda line: ((line[1],line[3]),line[2])) \
    .combineByKey(lambda value: (value, 1), \
    lambda x, value: (x[0] + value, x[1] + 1),\
    lambda x, y: (x[0] + y[0], x[1] + y[1])) \
    .map(lambda (label, (value_sum, count)): (label, value_sum / count)) \
    .map (lambda x: (x[0][0] , x[0][1], x[1]))
    
    userMovieRating = movieUserRatingCluster.map(lambda x: (x[1], x[0], x[2])).map(lambda x: ((x[0], x[1]), x[2]))
    
    clusterUserAvg = userClusterAvg.map(lambda x: (x[1], (x[0], x[2])))
    clusterMovie = movieCluster.map(lambda x: (x[1], x[0]))
    
    clusterUserAvgMovie = clusterUserAvg.leftOuterJoin(clusterMovie).map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][1]))
    
    userMovieAvg = clusterUserAvgMovie.map(lambda x: ((x[1], x[3]), x[2]))
    
    userMovieRatingAvg = userMovieAvg.join(userMovieRating)
    
    testRmse = sqrt(userMovieRatingAvg.map(lambda x: (x[1][0] - x[1][1]) ** 2).reduce(add) / float(numRatings))
    meanRating = ratings.map(lambda x: x[1][2]).mean()
    baselineRmse = sqrt(userMovieRatingAvg.map(lambda x: (meanRating - x[1][0]) ** 2).reduce(add) / numRatings)
    improvement = (baselineRmse - testRmse) / baselineRmse * 100
    print("The test RMSE is %f and the baseline RMSE is %f") % (testRmse, baselineRmse)
    print("The best model improves the baseline by %.2f" % (improvement) + "%.")
