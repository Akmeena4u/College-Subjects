#include <bits/stdc++.h>
using namespace std;

int replacement = -1;

// Structure to represent data points with attributes (m, n) and class label (c)
struct val {
	int m, n;
	string c;
	bool a = false, b = false; // Flags to indicate whether attributes (m, n) are missing
};

// Function to print the dataset with missing values
void print(val data[]) {
	cout << "The data is :- \n\n";
	for (int i = 0; i < 20; i++) {
		if (data[i].a && data[i].b) {
			cout << data[i].m << ' ' << data[i].n << ' ' << data[i].c << endl;
		} else if (data[i].a) {
			cout << data[i].m << " -- " << data[i].c << endl;
		} else {
			cout << "-- " << "-- " << data[i].c << endl;
		}
	}
}

// Method 3: Filling missing values with the mean of attributes belonging to the same class
void method3(val data[]) {
	cout << "Mean of attributes belonging to the same class: \n";
	val data1[20];

	// Calculate mean and count for each class
	int mean[4];
	int cnt[4];
	for (int i = 0; i < 4; i++) mean[i] = 0, cnt[i] = 0;
	int cls;

	for (int j = 0; j < 20; j++) {
		cls = (data[j].c[1] - '0' - 1);
		if (data[j].a == true) { mean[cls] += data[j].m; cnt[cls]++; }
		if (data[j].b == true) { mean[cls] += data[j].n; cnt[cls]++; }
	}

	// Calculate the mean for each class
	for (int i = 0; i < 4; i++) mean[i] /= cnt[i];

	cout << "The means of the classes are:\n";
	for (int i = 0; i < 4; i++)
		cout << ("c" + to_string(i + 1)) << " " << mean[i] << endl;

	// Fill missing values with class-wise mean
	for (int j = 0; j < 20; j++) {
		if (data[j].a == false) {
			data1[j].m = mean[(data[j].c[1] - '0' - 1)];
			data1[j].a = true;
		} else {
			data1[j].m = data[j].m;
			data1[j].a = data[j].a;
		}

		if (data[j].b == false) {
			data1[j].n = mean[(data[j].c[1] - '0' - 1)];
			data1[j].b = true;
		} else {
			data1[j].n = data[j].n;
			data1[j].b = data[j].b;
		}

		data1[j].c = data[j].c;
	}

	print(data1); cout << "\n\n";
}

// Method 2: Using a measure of central tendency (mean) for each attribute
void method2(val data[]) {
	cout << "Using a measure of central tendency for the attribute. \n";
	val data1[20];

	// Calculate mean and count for both attributes
	int mean1 = 0, mean2 = 0, cnt1 = 0, cnt2 = 0;
	for (int j = 0; j < 20; j++) {
		if (data[j].a == true) { mean1 += data[j].m; cnt1++; }
		if (data[j].b == true) { mean2 += data[j].n; cnt2++; }
	}

	// Calculate the mean for both attributes
	mean1 = mean1 / cnt1;
	mean2 = mean2 / cnt2;

	cout << "The mean for first partition is " << mean1 << '\n';
	cout << "The mean for second partition is " << mean2 << '\n';

	// Fill missing values with the corresponding attribute mean
	for (int j = 0; j < 20; j++) {
		if (data[j].a == false) {
			data1[j].m = mean1;
			data1[j].a = true;
		} else {
			data1[j].m = data[j].m;
			data1[j].a = data[j].a;
		}

		if (data[j].b == false) {
			data1[j].n = mean2;
			data1[j].b = true;
		} else {
			data1[j].n = data[j].n;
			data1[j].b = data[j].b;
		}

		data1[j].c = data[j].c;
	}

	print(data1); cout << "\n\n";
}

// Method 1: Replacing missing values with a global constant
void method1(val data[]) {
	cout << "Using the first method. i.e Replacing using a global constant\n\n";
	val data1[20];

	// Fill missing values with the global constant
	for (int j = 0; j < 20; j++) {
		if (data[j].a == false) {
			data1[j].m = replacement;
			data1[j].a = true;
		} else {
			data1[j].m = data[j].m;
			data1[j].a = data[j].a;
		}

		if (data[j].b == false) {
			data1[j].n = replacement;
			data1[j].b = true;
		} else {
			data1[j].n = data[j].n;
			data1[j].b = data[j].b;
		}

		data1[j].c = data[j].c;
	}

	print(data1); cout << "\n\n";
}

int main() {
	val data[20];
	srand(time(0));

	// Generate random data for attributes (m, n) and class labels (c)
	for (int j = 0; j < 20; j++) {
		if (rand() % 100 + 1 > 15) {
			data[j].m = rand() % 11 + 19; data[j].a = true;
		} else {
			data[j].a = false;
		}
		if (rand() % 100 + 1 > 20) {
			data[j].n = rand() % 13 + 15; data[j].b = true;
		} else {
			data[j].b = false;
		}
		data[j].c = ("c" + to_string(rand() % 4 + 1));
	}

	print(data);
	method1(data);
	method2(data);
	method3(data);

	return 0;
}
