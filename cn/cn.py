import networkx as nx

G = nx.Graph()

nn = int(input("Enter Number of Nodes: "))

nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H'][:nn]
G.add_nodes_from(nodes)

for i in range(nn):
    for j in range(i + 1, nn):
        dist = int(input(f"Enter Distance between {nodes[i]} and {nodes[j]}: "))
        G.add_edge(nodes[i], nodes[j], weight=dist)

print("\nShortest Paths and Distances:")
for i in range(nn):
    for j in range(i + 1, nn):
        path = nx.shortest_path(G, source=nodes[i], target=nodes[j], weight='weight')
        distance = nx.shortest_path_length(G, source=nodes[i], target=nodes[j], weight='weight')
        print(f"\nShortest path between {nodes[i]} and {nodes[j]}: {path}, Distance: {distance}")