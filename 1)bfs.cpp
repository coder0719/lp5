#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

// Graph represented using adjacency list
class Graph {
    int V; // Number of vertices
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int u, int v) {
        adj[u].push_back(v); // Undirected graph
        adj[v].push_back(u);
    }

    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        vector<int> frontier;
        frontier.push_back(start);
        visited[start] = true;

        cout << "Parallel BFS starting from node " << start << ":\n";

        while (!frontier.empty()) {
            vector<int> nextFrontier;

            // Parallel region
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
                            // Atomic check and set
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

                // Merge local frontiers into the global one
                #pragma omp critical
                nextFrontier.insert(nextFrontier.end(), localFrontier.begin(), localFrontier.end());
            }

            frontier = nextFrontier;
        }

        cout << endl;
    }
};

int main() {
    int V = 8; // Number of vertices
    Graph g(V);

    // Sample edges for undirected graph
    g.addEdge(0, 1);
    g.addEdge(0, 3);
    g.addEdge(1, 2);
    g.addEdge(3, 4);
    g.addEdge(4, 5);
    g.addEdge(2, 6);
    g.addEdge(5, 7);

    // Start BFS from vertex 0
    g.parallelBFS(0);

    return 0;
}
