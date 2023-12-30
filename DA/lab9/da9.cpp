#include <iostream>
#include <queue>
#include <vector>
#include <map>
#include <utility>
#include <climits>

typedef std::vector<std::map<int, int> > AdjList;
typedef unsigned long long int ULL;

struct TGraph {
    size_t n_ver;
    size_t n_edges;
    AdjList adj;
};

const ULL INF = ULLONG_MAX; 

void InputGraph(TGraph &graph) {
    int vertex_from, vertex_to, weight;
    graph.adj = AdjList(graph.n_ver + 1);
    
    for (int i = 0; i < graph.n_edges; i++) {
        std::cin >> vertex_from >> vertex_to >> weight;
        graph.adj[vertex_from][vertex_to] = weight;
        graph.adj[vertex_to][vertex_from] = weight;    
    }
}

ULL Dijkstra(TGraph &graph, size_t start, size_t finish) {
    std::vector<ULL> dist(graph.n_ver + 1, INF);
  
    dist[start] = 0;
    std::priority_queue< std::pair<ULL, int>, std::vector< std::pair<ULL, int> >, std::greater< std::pair<ULL, int> > > pq;
    pq.push (std::make_pair ((ULL)0, start));
	while (!pq.empty()) {
		int v = pq.top().second;
		ULL cur_d = pq.top().first;
		pq.pop();
		if (cur_d > dist[v])  continue;
 
		for (std::map<int, int>::iterator it = graph.adj[v].begin(); it != graph.adj[v].end(); it++) {
			int to = it->first, weight = it->second;
			//std::cout << "\t" << dist[v] << " " << weight << " " << dist[to] << std::endl;
			if (dist[v] + weight < dist[to]) {
				dist[to] = dist[v] + weight;
				pq.push (std::make_pair (dist[to], to));
			}
		}
		/*for(int i = 0; i < graph.n_ver + 1; i++)
		    std::cout << dist[i] << " ";
        std::cout << std::endl;*/			    
	}
	
    return dist[finish]; 
}

int main() {
    size_t start;
    size_t finish;
    ULL result = 0;
    TGraph graph;
    std::cin >> graph.n_ver >> graph.n_edges >> start >> finish;
    InputGraph(graph);
    
    result = Dijkstra(graph, start, finish);
    if (result == INF) {
        std::cout << "No solution" << std::endl;
    } else {
        std::cout << result << std::endl;
    }
    return 0;
}
