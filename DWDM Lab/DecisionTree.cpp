#include <iostream>
#include <vector>
#include <map>
#include <cmath>

struct Node {
    std::string attribute; // Attribute to split on
    std::string label;     // The predicted label if the node is a leaf
    std::vector<Node*> children;
};

double calculateEntropy(const std::vector<std::string>& labels) {
    std::map<std::string, int> labelCounts;
    for (const auto& label : labels) {
        labelCounts[label]++;
    }

    double entropy = 0.0;
    int totalLabels = labels.size();
    for (const auto& pair : labelCounts) {
        double probability = static_cast<double>(pair.second) / totalLabels;
        entropy -= probability * log2(probability);
    }

    return entropy;
}

double calculateInformationGain(const std::vector<std::string>& attributeValues,
                                const std::vector<std::string>& labels,
                                const std::string& attribute) {
    std::map<std::string, std::vector<std::string>> attributeToLabels;
    for (size_t i = 0; i < attributeValues.size(); ++i) {
        attributeToLabels[attributeValues[i]].push_back(labels[i]);
    }

    double totalEntropy = calculateEntropy(labels);
    double weightedEntropy = 0.0;

    for (const auto& pair : attributeToLabels) {
        double probability = static_cast<double>(pair.second.size()) / labels.size();
        double entropy = calculateEntropy(pair.second);
        weightedEntropy += probability * entropy;
    }

    return totalEntropy - weightedEntropy;
}

Node* buildDecisionTree(const std::vector<std::vector<std::string>>& dataset,
                        const std::vector<std::string>& attributes,
                        const std::vector<std::string>& labels) {
    Node* root = new Node;

    // If all labels are the same, return a leaf node
    if (std::all_of(labels.begin(), labels.end(), [&](const std::string& label) {
            return label == labels[0];
        })) {
        root->label = labels[0];
        return root;
    }

    // If no attributes left or only one unique label, return a leaf node with the majority label
    if (attributes.empty() || std::set<std::string>(labels.begin(), labels.end()).size() == 1) {
        std::map<std::string, int> labelCounts;
        for (const auto& label : labels) {
            labelCounts[label]++;
        }

        int maxCount = 0;
        std::string majorityLabel;
        for (const auto& pair : labelCounts) {
            if (pair.second > maxCount) {
                maxCount = pair.second;
                majorityLabel = pair.first;
            }
        }

        root->label = majorityLabel;
        return root;
    }

    // Find the attribute with the highest information gain
    double maxInformationGain = 0.0;
    std::string bestAttribute;
    for (const auto& attribute : attributes) {
        double informationGain = calculateInformationGain(dataset[0], labels, attribute);
        if (informationGain > maxInformationGain) {
            maxInformationGain = informationGain;
            bestAttribute = attribute;
        }
    }

    root->attribute = bestAttribute;

    // Create child nodes and recursively build the decision tree
    std::map<std::string, std::vector<std::vector<std::string>>> attributeToData;
    for (size_t i = 0; i < dataset[0].size(); ++i) {
        attributeToData[dataset[0][i]].push_back(dataset[i]);
    }

    std::vector<std::string> newAttributes(attributes.begin(), attributes.end());
    newAttributes.erase(std::remove(newAttributes.begin(), newAttributes.end(), bestAttribute),
                        newAttributes.end());

    for (const auto& pair : attributeToData) {
        Node* child = buildDecisionTree(pair.second, newAttributes, pair.second[0]);
        root->children.push_back(child);
    }

    return root;
}

void deleteTree(Node* root) {
    if (root) {
        for (Node* child : root->children) {
            deleteTree(child);
        }
        delete root;
    }
}

void printTree(Node* root, int depth = 0) {
    if (root) {
        for (int i = 0; i < depth; ++i) std::cout << "  ";
        if (!root->attribute.empty()) {
            std::cout << root->attribute << std::endl;
            for (Node* child : root->children) {
                printTree(child, depth + 1);
            }
        } else {
            std::cout << "Prediction: " << root->label << std::endl;
        }
    }
}

int main() {
    std::vector<std::string> attributes = {"Outlook", "Temperature", "Humidity", "Windy"};
    std::vector<std::vector<std::string>> golfDataset = {
        // ... (data from the golf dataset)
    };
    std::vector<std::string> labels = golfDataset.back();
    golfDataset.pop_back();

    Node* root = buildDecisionTree(golfDataset, attributes, labels);

    std::cout << "Decision Tree:" << std::endl;
    printTree(root);

    // Remember to delete the dynamically allocated tree nodes to avoid memory leaks
    deleteTree(root);
    return 0;
}
