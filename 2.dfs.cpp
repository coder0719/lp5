#include <iostream>
#include <vector>
#include <stack>
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
        adj[v].push_back(u); // undirected graph
    }

    void parallelDFS(int start) {
        vector<bool> visited(V, false);

        cout << "Parallel DFS starting from node " << start << ":\n";

        // Stack for DFS
        stack<int> s;
        s.push(start);

        while (!s.empty()) {
            int u = s.top();
            s.pop();

            if (!visited[u]) {
                visited[u] = true;
                cout << u << " ";

                // Parallelize exploration of neighbors
                #pragma omp parallel for
                for (int i = 0; i < adj[u].size(); ++i) {
                    int v = adj[u][i];
                    if (!visited[v]) {
                        #pragma omp critical
                        s.push(v);
                    }
                }
            }
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

    int start;
    cout << "Enter starting node: ";
    cin >> start;

    g.parallelDFS(start);

    return 0;
}
