// the Naive Bayes classifier is a simple and popular algorithm used for classification tasks.
// It is based on the Bayes theorem and assumes that the features are conditionally independent given the class label.

// The Naive Bayes algorithm works by calculating the probability of each class label given the features of a data point and 
// then selecting the class label with the highest probability

#include <bits/stdc++.h>
#include <fstream> 
using namespace std;
int main() {
    // Open the "golf.txt" file for reading
    ifstream fin("golf.txt");
    
    // Number of instances (n) and number of features (f) in the dataset
    int n = 14, f = 4;
    
    // Create a 2D vector to store the dataset with 'n' rows and 'f+1' columns
    // The additional column is for storing the target class label
    vector<vector<string>> v(n, vector<string>(f + 1));
    
    // Read the dataset from the file and store it in the vector 'v'
    for (int i = 0; i < n; i++)
        for (int j = 0; j < f + 1; j++)
            fin >> v[i][j];

    // Create a map 'mp' to store the count of each target class label in the dataset
    map<string, float> mp;
    for (int i = 0; i < n; i++)
        mp[v[i][f]]++;

    // Create a vector 'in' to store the input tuple for classification
    vector<string> in(f);
    int k = 0;
    cout << "Enter Outlook : ";
    cin >> in[k++];
    cout << "Enter Temperature : ";
    cin >> in[k++];
    cout << "Enter Humidity : ";
    cin >> in[k++];
    cout << "Enter Windy : ";
    cin >> in[k++];

    // Create a vector of pairs 'ans' to store the probability of each target class
    // The pair stores the class label and its corresponding probability
    vector<pair<string, float>> ans;
    for (auto itr = mp.begin(); itr != mp.end(); ++itr)
        ans.push_back({itr->first, 0});

    // Number of distinct target classes (c)
    int c = ans.size();

    // Create a 2D array 'F' to store the count of occurrences of feature values for each class
    float F[c][f];
    for (int i = 0; i < c; i++)
        for (int j = 0; j < f; j++)
            F[i][j] = 0;

    // Calculate the occurrences of feature values for each class in the dataset
    for (int i = 0; i < n; i++)
        for (int j = 0; j < c; j++)
            for (int k = 0; k < f; k++)
                if (in[k] == v[i][k] && v[i][f] == ans[j].first)
                    F[j][k]++;

    // Calculate the probabilities for each target class based on the input tuple 'in'
    for (int i = 0; i < c; i++) {
        ans[i].second = 1;
        for (int j = 0; j < f; j++)
            ans[i].second *= F[i][j];

        ans[i].second = ans[i].second / pow(mp[ans[i].first], f) * mp[ans[i].first] / n;
    }

    // Print the probabilities that the given tuple belongs to each class
    for (int i = 0; i < c; i++)
        cout << "Probability that given tuple belongs to class : " << ans[i].first << " " << ans[i].second << endl;

    // Find the class with the highest probability and output the result
    int mx = 0;
    for (int i = 1; i < c; i++)
        if (ans[i].second > ans[mx].second)
            mx = i;

    cout << "\nClass : " << ans[mx].first;
    return 0;
}



