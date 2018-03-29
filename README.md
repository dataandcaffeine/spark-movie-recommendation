# spark-movie-recommendation

Movie Recommendation Two Ways ALS and K-Means Predictions in PySpark 

# Run Programs 
1. Store u.data and u.item on Hadoop Distributed File System in a folder called input. 
2. Execute Alternating Least Squares recommendation using :  spark-submit --driver-memory 512m MovieLensALS.py hdfs://localhost:9000/input  3. Execute KMeans recommendation using: spark-submit --driver-memory 512m MovieLensKM.py hdfs://localhost:9000/input 
