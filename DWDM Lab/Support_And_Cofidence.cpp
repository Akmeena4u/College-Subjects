/*In data mining, support and confidence are two important metrics used in association rule mining to identify interesting patterns
and relationships between items in a dataset. They are often used in algorithms like the Apriori algorithm and FP-Growth to discover. 

Support-Support is the proportion of transactions that contain a particular itemset.
Confidence: Confidence measures the likelihood that an item B is purchased when item A is purchased.


Support(X) = (Number of transactions containing X) / (Total number of transactions in D)
Confidence(X → Y) = Support(X ∪ Y) / Support(X)


1. Calculate support of Milk, Beer, Diapers, Bread, Butter, Cookies.
Which of these have support greater than 0.3 ?
2. Calculate confidence :
 Cake -> Tea
 Tea -> Cake
 Milk->Tea
 Milk, Diaper -> Beer
 Milk -> Bread, Cookies
And state the rules if min confidence is 25%. 1 Milk, Beer, Diapers
2 Bread, Butter, Milk
3 Milk, Diapers, Cookies
4 Bread, Butter, Cookies
5 Beer, Cookies, Diapers
6 Milk, Diapers, Bread, Butter
7 Bread, Butter, Diapers
8 Beer, Diapers
9 Milk, Diapers, Bread, Butter
10 Beer, Cookies


  */

#include <bits/stdc++.h>
#include <fstream>
using namespace std;

int main()
{
    ifstream fin("data.txt"); // Open the input file for reading
    int n = 10; // Number of transactions
    vector<vector<string>> v(n); // Vector of vectors to store the transactions

    // Read the transactions from the input file
    for (int i = 0; i < n; i++)
    {
        string s;
        while (s != "n")
        {
            fin >> s;
            if (s != "n")
                v[i].push_back(s); // Push items into the transaction vector
        }
    }

    map<string, float> mp; // Map to store the count of each item
    for (int i = 0; i < n; i++)
    {
        for (int j = 1; j < v[i].size(); j++)
            mp[v[i][j]]++; // Increment the count for each item in the map
    }

    map<string, float>::iterator itr;
    vector<string> allowed; // Vector to store items with support greater than 0.3
    cout << "\nSupport of - \n";
    for (itr = mp.begin(); itr != mp.end(); itr++)
    {
        float support = itr->second / n; // Calculate the support of the item
        cout << itr->first << " : " << support << endl;
        if (support > 0.3)
            allowed.push_back(itr->first); // If support > 0.3, add to allowed vector
    }

    cout << "\nThe items that have support greater than given threshold (30%) are: \n";
    for (int i = 0; i < allowed.size(); i++)
        cout << allowed[i] << " ";

    int K;
    cout << "\n\nEnter the number of checks : ";
    cin >> K;

    // Loop to check confidence for K rules entered by the user
    for (int z = 0; z < K; z++)
    {
        vector<string> X, Y;

        cout << "\nEnter X : ";
        string s;
        while (cin >> s && s != "-1")
            X.push_back(s); // Input the antecedent (left-hand side) of the rule
        cout << "Enter Y : ";
        while (cin >> s && s != "-1")
            Y.push_back(s); // Input the consequent (right-hand side) of the rule

        float a = 0, b = 0;
        for (int i = 0; i < n; i++)
        {
            int x = 0, y = 0;
            for (int j = 0; j < X.size(); j++)
            {
                for (int k = 1; k < v[i].size(); k++)
                {
                    if (X[j] == v[i][k])
                    {
                        x++; // Count occurrences of antecedent in the transaction
                        break;
                    }
                }
            }

            if (x == X.size())
                b++; // Count transactions containing the antecedent (X)

            if (x == X.size())
                for (int j = 0; j < Y.size(); j++)
                {
                    for (int k = 1; k < v[i].size(); k++)
                    {
                        if (Y[j] == v[i][k])
                        {
                            y++; // Count occurrences of consequent in the transaction
                            break;
                        }
                    }
                }

            if (y == Y.size() && x == X.size())
                a++; // Count transactions containing both X and Y (antecedent and consequent)
        }

        float m;
        if (b > 0)
            m = a / b; // Calculate confidence (support of both X and Y / support of X)
        else
            m = 0;

        cout << "Confidence : " << m;
        if (m >= 0.25)
            cout << "\nEntered rule is acceptable"; // Check if the confidence meets the threshold
        cout << endl;
    }

    return 0;
}
