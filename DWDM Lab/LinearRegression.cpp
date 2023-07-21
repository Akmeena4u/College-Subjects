/*
   Write a program to calculate values of regression coefficients a and b in equation y=ax+b . Consider 5-10 points.
Calculate:
1. a and b
2. sum of squared error
3. Plot the line using any simple pattern (*) on terminal.
*/


#include<bits/stdc++.h>
using namespace std;

bool comp(pair<float, float> a, pair<float, float> b) {
    return b.second < a.second;
}

int main() {
    float n, sumx = 0, sumy = 0, sumxy = 0, sumx2 = 0;
    cin >> n;
    vector<pair<float, float>> vec(n);
    for (int i = 0; i < n; i++) {
        cin >> vec[i].first >> vec[i].second;
        sumx += vec[i].first;
        sumy += vec[i].second;
        sumxy += (vec[i].first * vec[i].second);
        sumx2 += (vec[i].first * vec[i].first);
    }

    float a = n * sumxy - (sumy * sumx);
    a /= (n * sumx2) - (sumx * sumx);
    float b = sumy - (a * sumx);
    b /= n;

    cout << "Value of a: " << a << endl;
    cout << "Value of b: " << b << endl;

    float error = 0;
    for (int i = 0; i < n; i++) {
        error += pow((vec[i].second - ((a * vec[i].first) + b)), 2);
    }
    cout << "Sum of squared error for the chosen optimal line: " << error << endl;

  

    return 0;
}
