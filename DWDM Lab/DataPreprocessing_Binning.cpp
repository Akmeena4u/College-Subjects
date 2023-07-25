/*
   Binning is a data preprocessing technique used to convert continuous numerical data into discrete intervals or bins. 
  */



#include <bits/stdc++.h>
using namespace std;

int main() {
    srand(time(0));
    int n = 12;
    float arr[n];
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 20 + 19; // Filling 'arr' with random values in the range [19, 38]
    }

    sort(arr, arr + n); // Sorting 'arr' in ascending order

    int binSize = 3;
    float arr1[n];

    // Displaying the original sorted data
    for (int i = 0; i < n; i++)
        cout << arr[i] << ' ';
    cout << "\n\n";

    // Method 1 of binning: Smoothing by bin-means
    cout << "Method1 of binning: Smoothing by bin-means: \n";

    vector<float> binMeans(n / binSize, 0);

    // Calculating bin means
    for (int i = 0; i < n; i++) {
        binMeans[i / binSize] += arr[i];
    }
    for (auto& i : binMeans) {
        i /= binSize;
    }
    for (int i = 0; i < n; i++) {
        arr1[i] = binMeans[i / binSize]; // Smoothing data using bin means
    }

    // Displaying the data after smoothing using bin means
    for (int i = 0; i < n; i++)
        cout << fixed << setprecision(1) << arr1[i] << ' ';
    cout << "\n\n";

    // Method 2 of binning: Smoothing by bin-median
    cout << "Method2 of binning: Smoothing by bin-median: \n";

    vector<int> binMedians(n / binSize, 0);
    for (int i = 0; i < n; i++) {
        binMedians[i / binSize] = arr[(i / binSize) * binSize + binSize / 2];
    }
    for (int i = 0; i < n; i++) {
        arr1[i] = binMedians[i / binSize]; // Smoothing data using bin medians
    }
    for (int i = 0; i < n; i++)
        cout << arr1[i] << ' ';
    cout << "\n\n";

    // Method 3 of binning: Smoothing by bin boundaries
    cout << "Method3 of binning: Smoothing by bin boundaries: \n";

    vector<vector<int>> binBounds(n / binSize, vector<int>(2, 0));
    for (int i = 0; i < n; i += binSize) {
        binBounds[i / binSize][0] = arr[i];
        binBounds[i / binSize][1] = arr[i + binSize - 1];
    }
    for (int i = 0; i < n; i++) {
        int min = binBounds[i / binSize][0];
        int max = binBounds[i / binSize][1];

        if (abs(arr[i] - min) <= abs(arr[i] - max))
            arr1[i] = min;
        else
            arr1[i] = max; // Smoothing data using bin boundaries (closest boundary value)
    }

    for (int i = 0; i < n; i++)
        cout << arr1[i] << ' ';
    cout << "\n\n";
}
