#include <iostream>
#include <vector>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // For undirected graph
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        vector<int> frontier;
        frontier.push_back(start);
        visited[start] = true;

        cout << "Parallel BFS starting from node " << start << ":\n";

        while (!frontier.empty()) {
            vector<int> nextFrontier;

            #pragma omp parallel
            {
                vector<int> localFrontier;

                #pragma omp for nowait
                for (int i = 0; i < frontier.size(); ++i) {
                    int u = frontier[i];
                    #pragma omp critical
                    cout << u << " ";

                    for (int v : adj[u]) {
                        if (!visited[v]) {
                            #pragma omp critical
                            {
                                if (!visited[v]) {
                                    visited[v] = true;
                                    localFrontier.push_back(v);
                                }
                            }
                        }
                    }
                }

                #pragma omp critical
                nextFrontier.insert(nextFrontier.end(), localFrontier.begin(), localFrontier.end());
            }

            frontier = nextFrontier;
        }

        cout << endl;
    }
};

int main() {
    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;

    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> E;

    cout << "Enter edges (u v) each in a new line (0-based indexing):\n";
    for (int i = 0; i < E; ++i) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int startNode;
    cout << "Enter the starting node for BFS: ";
    cin >> startNode;

    g.parallelBFS(startNode);

    return 0;
}
