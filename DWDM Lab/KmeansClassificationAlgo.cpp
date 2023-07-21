/*
k-means is a technique for data clustering that may be used for unsupervised machine learning.
It is capable of classifying unlabeled data into a predetermined number of clusters based on similarities (k).

In this method, data points are assigned to clusters in such a way that the sum of the squared distances between the data points and
the centroid is as small as possible. It is essential to note that reduced diversity within clusters leads to more identical data points within the same cluster.

K-means clustering algorithm steps in short:

1. Choose the number of clusters (K).
2. Initialize k cluster centroids randomly.
3. Assign data points to the nearest centroid.
4. Update centroids based on assigned data points.
5. Repeat steps 3 and 4 until convergence.
6. Final result: Data points are clustered into K groups.

*/

/*
    Take 20-30 records(can take 2 attributes)
Implement k-means algorithm
Take k value from user
Test the results for minimum 5 distinct values of k

*/

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Data point representation with two attributes X and Y
struct DataPoint {
    double X;
    double Y;
};

// Function to calculate the Euclidean distance between two data points
double distance(DataPoint p1, DataPoint p2) {
    return sqrt(pow(p1.X - p2.X, 2) + pow(p1.Y - p2.Y, 2));
}

// Function to find the index of the nearest centroid for a data point
int findNearestCentroid(DataPoint data, vector<DataPoint>& centroids) {
    double minDist = distance(data, centroids[0]);
    int nearestIndex = 0;

    for (int i = 1; i < centroids.size(); i++) {
        double dist = distance(data, centroids[i]);
        if (dist < minDist) {
            minDist = dist;
            nearestIndex = i;
        }
    }

    return nearestIndex;
}

// Function to update centroids based on assigned data points
void updateCentroids(vector<DataPoint>& data, vector<DataPoint>& centroids, vector<int>& clusters) {
    vector<int> counts(centroids.size(), 0);

    for (int i = 0; i < data.size(); i++) {
        int cluster = clusters[i];
        centroids[cluster].X += data[i].X;
        centroids[cluster].Y += data[i].Y;
        counts[cluster]++;
    }

    for (int i = 0; i < centroids.size(); i++) {
        centroids[i].X /= counts[i];
        centroids[i].Y /= counts[i];
    }
}

// K-means clustering function
vector<int> kMeansClustering(vector<DataPoint>& data, int K) {
    // Initialize random centroids
    vector<DataPoint> centroids(K);
    for (int i = 0; i < K; i++) {
        centroids[i] = data[rand() % data.size()];
    }

    vector<int> clusters(data.size());

    bool isConverged = false;
    while (!isConverged) {
        // Assign data points to the nearest centroids
        for (int i = 0; i < data.size(); i++) {
            clusters[i] = findNearestCentroid(data[i], centroids);
        }

        // Store old centroids to check for convergence
        vector<DataPoint> oldCentroids = centroids;

        // Update centroids based on assigned data points
        updateCentroids(data, centroids, clusters);

        // Check for convergence
        isConverged = true;
        for (int i = 0; i < centroids.size(); i++) {
            if (distance(oldCentroids[i], centroids[i]) > 0.001) {
                isConverged = false;
                break;
            }
        }
    }

    return clusters;
}

int main() {
    int n;
    cout << "Enter the number of data points: ";
    cin >> n;

    vector<DataPoint> data(n);
    cout << "Enter " << n << " data points (X Y): " << endl;
    for (int i = 0; i < n; i++) {
        cin >> data[i].X >> data[i].Y;
    }

    int k;
    cout << "Enter the value of K: ";
    cin >> k;

    srand(static_cast<unsigned int>(time(nullptr)));

    // Test for minimum 5 distinct values of K
    for (int i = k; i < k + 5; i++) {
        vector<int> clusters = kMeansClustering(data, i);
        cout << "K = " << i << endl;
        for (int j = 0; j < data.size(); j++) {
            cout << "(" << data[j].X << ", " << data[j].Y << ") - Cluster " << clusters[j] << endl;
        }
        cout << "-------------------------------" << endl;
    }

    return 0;
}







/*
Explanation---
Sure! Let's explain the code step by step in simple language:

1. We include necessary libraries (iostream, vector, cmath, cstdlib, ctime) to work with input/output, arrays, mathematical operations, and random number generation.

2. We define a struct `DataPoint` to represent a data point in two-dimensional space, containing attributes `X` and `Y`.

3. We define a function `distance` to calculate the Euclidean distance between two data points using their `X` and `Y` attributes.

4. We define a function `findNearestCentroid` to find the index of the nearest centroid for a given data point from a set of centroids. It uses the `distance` function to calculate distances.

5. We define a function `updateCentroids` to update the centroids based on the assigned data points. It recalculates the centroids' positions based on the mean of the data points assigned to each cluster.

6. We define the main function `kMeansClustering`, which implements the K-means clustering algorithm. It takes the data points and the number of clusters 'K' as input and returns a vector containing the cluster assignments for each data point.

7. In the `main` function, we input the number of data points and their coordinates (X and Y) from the user. We also take the value of 'K' (the number of clusters) as input.

8. We use the `rand()` function along with `srand(time(nullptr))` to generate random centroids for K-means initialization.

9. We test the K-means algorithm for K, K+1, K+2, K+3, and K+4 to explore its performance with different numbers of clusters.

10. For each value of K, we print the data points along with their respective cluster assignments.

11. The program outputs the results for different values of K, showing the clustering of data points into different clusters based on their proximity to the centroids.

  */
