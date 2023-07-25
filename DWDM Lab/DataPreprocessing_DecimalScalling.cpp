// The method involves shifting the decimal point of each data value by a fixed number of positions to achieve normalization.
// The goal is to bring the data values within a specific range,
// typically between -1 and 1, without altering the order or distribution of the data.

#include<bits/stdc++.h>
using namespace std;

int main() {
	// Input the number of data points 'n'
	int n;
	cin >> n;
	
	// Create an array 'arr' to store the input data
	float arr[n];
	
	// Input the data points
	for(int i = 0; i < n; i++) {
		cin >> arr[i];
	}
	
	// Find the maximum absolute value (maxval) among all data points
	float maxval = (float)INT_MIN;
	for(int i = 0; i < n; i++) {
		maxval = max(maxval, abs(arr[i]));
	}
	
	// Determine the scaling factor 't' by finding the smallest power of 10
	// such that maxval/t is greater than 1 (ensuring the normalized values
	// fall within the desired range)
	int t = 1;
	while(maxval/t > 1) {
		t *= 10;
	}
	
	// Normalize the data by dividing each data point by the scaling factor 't'
	for(int i = 0; i < n; i++) {
		cout << arr[i]/t << " ";
	}
	
	return 0;
}
