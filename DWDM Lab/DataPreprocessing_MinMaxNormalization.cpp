/*
MinMax normalization, also known as feature scaling
The normalization process transforms the original data values into a new range, 
typically between 0 and 1, making the data more comparable and avoiding the dominance of large magnitude values.


*/

#include<bits/stdc++.h>
using namespace std;

int main() {
	// Generate a random number of data points between 8 and 17
	int n = rand() % 10 + 8;

	// Create an array to hold the random data points between 17 and 65
	float arr[n];
	for (int i = 0; i < n; i++) {
		arr[i] = rand() % 49 + 17;
	}

	// Find the minimum and maximum values in the array
	float mn = (float)INT_MAX;
	float mx = (float)INT_MIN;
	for (int i = 0; i < n; i++) {
		mn = min(mn, arr[i]);
		mx = max(mx, arr[i]);
	}

	// Generate new_min and new_max for normalization (between 1 and 8)
	int new_mx = 0, new_mn = 0;
	while (new_mx == new_mn) {
		new_mx = rand() % 8 + 1;
		new_mn = rand() % 8 + 1;
		if (new_mx == new_mn) continue;
		else {
			// Ensure new_mx is greater than new_mn
			if (new_mx < new_mn) swap(new_mx, new_mn);
			break;
		}
	}

	cout << "Values: \n";
	// Display the original random data
	for (int i = 0; i < n; i++) {
		cout << arr[i] << " ";
	}
	cout << endl;

	// Normalize the data using the Min-Max normalization formula
	for (int i = 0; i < n; i++) {
		float t = arr[i];
		arr[i] = (t - mn) * (new_mx - new_mn) / (mx - mn) + new_mn;
	}

	cout << "Normalized: \n";
	// Display the normalized data
	for (int i = 0; i < n; i++) {
		cout << arr[i] << " ";
	}
	cout << endl;
	
	return 0;
}
