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

//Linear Refression for three variables 

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Function to perform linear regression and find the coefficients a, b, and c
void linearRegression(vector<double>& x, vector<double>& y, vector<double>& z, double& a, double& b, double& c) {
    int n = x.size();

    // Calculate the sum of x, y, z, x^2, y^2, xy, xz, yz
    double sum_x = 0, sum_y = 0, sum_z = 0, sum_x2 = 0, sum_y2 = 0, sum_xy = 0, sum_xz = 0, sum_yz = 0;

    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_z += z[i];
        sum_x2 += pow(x[i], 2);
        sum_y2 += pow(y[i], 2);
        sum_xy += x[i] * y[i];
        sum_xz += x[i] * z[i];
        sum_yz += y[i] * z[i];
    }

    // Calculate the coefficients a, b, and c
    double denominator = n * sum_x2 * sum_y2 + 2 * sum_x * sum_y * sum_xy - n * pow(sum_xy, 2) - pow(sum_x, 2) * sum_y2 - pow(sum_y, 2) * sum_x2;
    a = (sum_x2 * sum_yz + sum_y * sum_xz + sum_x * sum_xy - n * sum_xz * sum_y - sum_x * sum_yz - sum_xy * sum_x2) / denominator;
    b = (n * sum_xy * sum_z + sum_x * sum_yz + sum_y * sum_xz - sum_xz * sum_y2 - sum_xy * sum_yz - n * sum_x * sum_z) / denominator;
    c = (sum_z - a * sum_x - b * sum_y) / n;
}

// Function to calculate the value of z for a given x and y using the regression line
double calculateZ(double x, double y, double a, double b, double c) {
    return a * x + b * y + c;
}

int main() {
    // Given dataset of x, y, and z
    vector<double> x = {1.0, 2.0, 3.0, 4.0, 5.0};
    vector<double> y = {2.0, 3.0, 4.0, 5.0, 6.0};
    vector<double> z = {3.0, 4.0, 5.0, 6.0, 7.0};

    double a, b, c;
    
    // Perform linear regression and find the coefficients a, b, and c
    linearRegression(x, y, z, a, b, c);

    // Display the coefficients a, b, and c
    cout << "Coefficients: a = " << a << ", b = " << b << ", c = " << c << endl;

    // Display the regression line equation
    cout << "Regression line equation: z = " << a << "x + " << b << "y + " << c << endl;

    // Example: Calculate the value of z for x = 6.0 and y = 7.0 using the regression line
    double x_input = 6.0;
    double y_input = 7.0;
    double z_output = calculateZ(x_input, y_input, a, b, c);
    cout << "For x = " << x_input << " and y = " << y_input << ", z = " << z_output << endl;

    return 0;
}
