#include<bits/stdc++.h>
using namespace std;

// Function to calculate the mean of the data
float meanOfdata(float arr[], int size) {
	float sum = 0;
	// Calculate the sum of all elements in the array
	for(int i = 0; i < size; i++) {
		sum += arr[i];
	}
	// Return the mean by dividing the sum by the size of the array
	return sum / size;
}

// Function to calculate the standard deviation of the data
float sdOfdata(float arr[], int size, int mean) {
	float sum = 0;
	// Calculate the sum of squared differences between each data point and the mean
	for(int i = 0; i < size; i++) {
		sum += pow(abs(mean - arr[i]), 2);
	}
	// Return the standard deviation by taking the square root of the mean of squared differences
	return sqrt(sum / size);
}

int main() {
	// Z-score normalization
	int n;
	// Input the number of data points 'n'
	cin >> n;

	// Create an array 'arr' to store the input data
	float arr[n];

	// Input the data points
	for(int i = 0; i < n; i++) {
		cin >> arr[i];
	}

	// Calculate the mean of the data using the meanOfdata function
	float mean = meanOfdata(arr, n);

	// Calculate the standard deviation of the data using the sdOfdata function
	float sd = sdOfdata(arr, n, mean);

	// Perform Z-score normalization on the data by subtracting the mean from each data point
	// and dividing by the standard deviation
	for(int i = 0; i < n; i++) {
		arr[i] = (arr[i] - mean) / sd;
	}

	// Display the normalized data
	for(int i = 0; i < n; i++) {
		cout << arr[i] << " ";
	}

	return 0;
}
