/*
K-nearest neighbors (KNN) is a simple and widely used classification algorithm. In this algorithm, 
a data point is classified based on the majority class of its K-nearest neighbors in the feature space.

```cpp

*/
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
using namespace std;

// Function to calculate Euclidean distance between two data points
float euclideanDistance(const vector<float>& a, const vector<float>& b) {
    float distance = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        distance += pow(a[i] - b[i], 2);
    }
    return sqrt(distance);
}

int main() {
    ifstream fin;
    fin.open("iris.txt");

    // Number of data points (samples), features (attributes), and value of k (number of neighbors)
    int n = 30, f = 4, k;

    // Store the class labels and features of data points
    vector<string> C(n);
    vector<vector<float>> v(n, vector<float>(f));

    // Read data from the file and store in vectors
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < f; j++)
            fin >> v[i][j];
        fin >> C[i];
    }
    fin.close();

    cout << "\nFor Sample Enter -\n";
    vector<float> sample(f);
    cout << "Enter Sepal-Length : ";
    cin >> sample[0];
    cout << "Enter Sepal-Width : ";
    cin >> sample[1];
    cout << "Enter Petal-Length : ";
    cin >> sample[2];
    cout << "Enter Petal-Width : ";
    cin >> sample[3];

    cout << "\nEnter the value of k : ";
    cin >> k;

    vector<pair<float, string>> distances;
    for (int i = 0; i < n; i++) {
        // Calculate Euclidean distance between the sample and all data points
        float distance = euclideanDistance(sample, v[i]);
        distances.push_back({distance, C[i]});
    }

    // Sort the distances in ascending order
    sort(distances.begin(), distances.end());

    map<string, int> labelFrequency;
    map<string, float> labelDistance;

    for (int i = 0; i < k; i++) {
        // Count the occurrences of each label in the k-nearest neighbors
        labelFrequency[distances[i].second]++;
        // Calculate the sum of distances for each label in the k-nearest neighbors
        labelDistance[distances[i].second] += distances[i].first;
    }

    vector<pair<int, string>> topk;
    for (const auto& itr : labelFrequency)
        topk.push_back({itr.second, itr.first});

    // Sort the labels by their frequencies in descending order
    sort(topk.begin(), topk.end(), greater<pair<int, string>>());

    cout << "\nFrequency of given sample belongs to :\n";
    for (const auto& itr : topk)
        cout << itr.second << " is : " << itr.first << endl;

    // Find the label with the highest frequency, breaking ties using the sum of distances
    int j = topk.size();
    int mx = topk[j - 1].first, ansi = j - 1;
    float V = labelDistance[topk[j - 1].second];
    j--;
    while (j >= 1 && topk[j - 1].first == mx) {
        if (labelDistance[topk[j - 1].second] < V) {
            V = labelDistance[topk[j - 1].second];
            ansi = j - 1;
        }
        j--;
    }

    cout << "\nHence, given sample belongs to class : " << topk[ansi].second;

    return 0;
}
```
/*
Certainly! Here's a summary of the steps performed in the given C++ code for the K-nearest neighbors (KNN) algorithm on the Iris dataset:

1. Read the data from the "iris.txt" file, which contains the Iris dataset. The data consists of four features (sepal length, sepal width, petal length, petal width) and the corresponding species labels.
2. Define a function `euclideanDistance` that calculates the Euclidean distance between two data points represented as vectors of floating-point numbers. It uses the formula for Euclidean distance: `sqrt((a[0] - b[0])^2 + (a[1] - b[1])^2 + ... + (a[n-1] - b[n-1])^2)`.
3. In the `main` function:
    - Open the "iris.txt" file and read the data into vectors for features and class labels.
    - Prompt the user to enter the feature values for an unknown sample for which we want to determine the species.
    - Ask the user to enter the value of `k`, which is the number of nearest neighbors to consider for classification.
4. Calculate the Euclidean distances between the unknown sample and all the data points in the dataset.
5. Sort the distances in ascending order to find the k-nearest neighbors.
6. Create two maps: `labelFrequency` to store the frequency of each class label among the k-nearest neighbors, and `labelDistance` to store the sum of distances for each class label among the k-nearest neighbors.
7. For the k-nearest neighbors, count the occurrences of each class label in `labelFrequency`, and calculate the sum of distances for each class label in `labelDistance`.
8. Create a vector `topk` to store the class labels along with their frequencies.
9. Sort the class labels in `topk` by their frequencies in descending order using `sort` with the `greater` comparator.
10. Print the frequencies of each class label among the k-nearest neighbors for the unknown sample.
11. Determine the class label with the highest frequency (`mx`) among the k-nearest neighbors. In case of a tie, choose the class label with the smallest sum of distances (`V`).
12. Print the predicted class label for the unknown sample based on the majority vote among the k-nearest neighbors.

This code allows you to classify an unknown sample from the Iris dataset based on its k-nearest neighbors using the KNN algorithm.  */
