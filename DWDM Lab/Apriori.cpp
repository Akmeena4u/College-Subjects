#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

using namespace std;

// Function to find the frequent itemsets that meet the minimum support threshold
unordered_map<string, int> findFrequentItemsets(vector<vector<string>>& transactions, double minSupport) {
    unordered_map<string, int> itemCounts;
    int nTransactions = transactions.size();

    // Count the occurrences of each item in the transactions
    for (const auto& transaction : transactions) {
        for (const auto& item : transaction) {
            itemCounts[item]++;
        }
    }

    // Find frequent items that meet the minimum support threshold
    unordered_map<string, int> frequentItems;
    for (const auto& pair : itemCounts) {
        double support = static_cast<double>(pair.second) / nTransactions;
        if (support >= minSupport) {
            frequentItems[pair.first] = pair.second;
        }
    }

    return frequentItems;
}

// Function to generate candidate itemsets of size k from frequent itemsets of size k-1
vector<vector<string>> generateCandidates(vector<vector<string>>& frequentItemsets, int k) {
    vector<vector<string>> candidates;

    int n = frequentItemsets.size();
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            vector<string> candidate;

            // Merge two frequent itemsets
            for (const auto& item : frequentItemsets[i]) {
                candidate.push_back(item);
            }
            for (const auto& item : frequentItemsets[j]) {
                candidate.push_back(item);
            }

            // Sort the candidate itemset to remove duplicates
            sort(candidate.begin(), candidate.end());

            // Add the candidate itemset to the list
            candidates.push_back(candidate);
        }
    }

    return candidates;
}

// Function to check if a transaction contains a given itemset
bool transactionContains(const vector<string>& transaction, const vector<string>& itemset) {
    for (const auto& item : itemset) {
        if (find(transaction.begin(), transaction.end(), item) == transaction.end()) {
            return false;
        }
    }
    return true;
}

// Function to find the frequent itemsets that meet the minimum support threshold
vector<vector<string>> apriori(vector<vector<string>>& transactions, double minSupport) {
    vector<vector<string>> frequentItemsets;
    int k = 1;

    // Find frequent itemsets of size 1
    frequentItemsets = findFrequentItemsets(transactions, minSupport);

    // Continue to find frequent itemsets of size k until no more frequent itemsets are found
    while (!frequentItemsets.empty()) {
        vector<vector<string>> candidates = generateCandidates(frequentItemsets, k);

        // Count occurrences of each candidate itemset in the transactions
        unordered_map<string, int> candidateCounts;
        for (const auto& transaction : transactions) {
            for (const auto& candidate : candidates) {
                if (transactionContains(transaction, candidate)) {
                    candidateCounts[vectorToString(candidate)]++;
                }
            }
        }

        // Find frequent itemsets that meet the minimum support threshold
        frequentItemsets.clear();
        for (const auto& pair : candidateCounts) {
            double support = static_cast<double>(pair.second) / transactions.size();
            if (support >= minSupport) {
                frequentItemsets.push_back(stringToVector(pair.first));
            }
        }

        k++;
    }

    return frequentItemsets;
}

int main() {
    // Given dataset
    vector<vector<string>> transactions = {
        {"Milk", "Beer", "Diapers"},
        {"Bread", "Butter", "Milk"},
        {"Milk", "Diapers", "Cookies"},
        {"Bread", "Butter", "Cookies"},
        {"Beer", "Cookies", "Diapers"},
        {"Milk", "Diapers", "Bread", "Butter"},
        {"Bread", "Butter", "Diapers"},
        {"Beer", "Diapers"},
        {"Milk", "Diapers", "Bread", "Butter"},
        {"Beer", "Cookies"}
    };

    double minSupport = 0.3; // Minimum support threshold

    // Find frequent itemsets using Apriori algorithm
    vector<vector<string>> frequentItemsets = apriori(transactions, minSupport);

    // Display the frequent itemsets
    cout << "Frequent Itemsets:\n";
    for (const auto& itemset : frequentItemsets) {
        for (const auto& item : itemset) {
            cout << item << " ";
        }
        cout << endl;
    }

    return 0;
}
