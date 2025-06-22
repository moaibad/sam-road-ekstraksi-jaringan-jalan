import pickle
import networkx as nx
import matplotlib.pyplot as plt
import scipy
from roadtracer.lib.discoverlib import geom
from roadtracer.lib.discoverlib import graph
from roadtracer.lib.discoverlib import rdp
import imageio.v2 as imageio
import numpy
import random
from io import StringIO

# MIN_GRAPH_DISTANCE = 16.6
# MAX_STRAIGHT_DISTANCE = 8.3
# MIN_GRAPH_DISTANCE = 166
# MAX_STRAIGHT_DISTANCE = 83
RDP_EPSILON = 2

def get_connections(g, im, min_graph_distance, max_straight_distance, limit=None):
    edge_im = -numpy.ones(im.shape, dtype='int32')
    for edge in g.edges:
        for p in geom.draw_line(edge.src.point, edge.dst.point, geom.Point(edge_im.shape[0], edge_im.shape[1])):
            edge_im[p.x-1, p.y-1] = edge.id
    
    road_segments, _ = graph.get_graph_road_segments(g)
    random.shuffle(road_segments)
    best_rs = None
    seen_vertices = set()
    proposed_connections = []
    for rs in road_segments:
        for vertex, opp in [(rs.src(), rs.point_at_factor(10)), (rs.dst(), rs.point_at_factor(rs.length() - 10))]:
            if len(vertex.out_edges) >= 2 or vertex in seen_vertices:
                continue
            seen_vertices.add(vertex)

            vertex_distances = get_vertex_distances(vertex, min_graph_distance)
            edge, path = get_shortest_path(im, vertex.point, opp, edge_im, g, vertex_distances, min_graph_distance, max_straight_distance)
            if edge is not None:
                proposed_connections.append({
                    'src': vertex.id,
                    'edge': edge.id,
                    'pos': edge.closest_pos(path[-1]).distance,
                    'path': rdp.rdp([(p.x, p.y) for p in path], RDP_EPSILON),
                })
        if limit is not None and len(proposed_connections) >= limit:
            break

    return proposed_connections

def insert_connections(g, connections):
    split_edges = {}  # map from edge to (split pos, new edge before pos, new edge after pos)
    
    for idx, connection in enumerate(connections):
        # figure out which current edge the connection intersects
        edge = g.edges[connection['edge']]
        path = [geom.Point(p[0], p[1]) for p in connection['path']]
        intersection_point = path[-1]
        
        while edge in split_edges:
            our_pos = edge.closest_pos(intersection_point).distance
            if our_pos < split_edges[edge]['pos']:
                edge = split_edges[edge]['before']
            else:
                edge = split_edges[edge]['after']

        # add path vertices
        prev_vertex = g.vertices[connection['src']]
        for point in path[1:]:
            vertex = g.add_vertex(point)
            edge1, edge2 = g.add_bidirectional_edge(prev_vertex, vertex)
            edge1.phantom = True
            edge1.connection_idx = idx
            edge2.phantom = True
            edge2.connection_idx = idx
            prev_vertex = vertex

        # split the edge
        new_vertex = prev_vertex
        
        for edge in [edge, edge.get_opposite_edge()]: 
            if edge is None:
                continue  # Skip processing if the edge is None
            
            split_pos = edge.closest_pos(intersection_point).distance
            split_edges[edge] = {
                'pos': split_pos,
                'before': g.add_edge(edge.src, new_vertex),
                'after': g.add_edge(new_vertex, edge.dst),
            }

    # remove extraneous edges
    filter_edges = set([edge for edge in split_edges.keys()])
    g = g.filter_edges(filter_edges)
    
    return g

def get_vertex_distances(src, max_distance):
	vertex_distances = {}

	seen_vertices = set()
	distances = {}
	distances[src] = 0
	while len(distances) > 0:
		closest_vertex = None
		closest_distance = None
		for vertex, distance in distances.items():
			if closest_vertex is None or distance < closest_distance:
				closest_vertex = vertex
				closest_distance = distance

		del distances[closest_vertex]
		vertex_distances[closest_vertex] = closest_distance
		seen_vertices.add(closest_vertex)
		if closest_distance > max_distance:
			break

		for edge in closest_vertex.out_edges:
			vertex = edge.dst
			if hasattr(edge, 'cost'):
				distance = closest_distance + edge.cost
			else:
				distance = closest_distance + edge.segment().length()
			if vertex not in seen_vertices and (vertex not in distances or distance < distances[vertex]):
				distances[vertex] = distance

	return vertex_distances

def get_shortest_path(im, src, opp, edge_im, g, vertex_distances, min_graph_distance, max_straight_distance):
	r = src.bounds().add_tol(max_straight_distance)
	r = geom.Rectangle(geom.Point(0, 0), geom.Point(im.shape[0], im.shape[1])).clip_rect(r)
	seen_points = set()
	distances = {}
	prev = {}
	dst_edge = None
	dst_point = None

	distances[src] = 0
	while len(distances) > 0:
		closest_point = None
		closest_distance = None
		for point, distance in distances.items():
			if closest_point is None or distance < closest_distance:
				closest_point = point
				closest_distance = distance

		del distances[closest_point]
		seen_points.add(closest_point)
		if edge_im[closest_point.x-1, closest_point.y-1] >= 0:
			edge = g.edges[edge_im[closest_point.x-1, closest_point.y-1]]
			src_distance = vertex_distances.get(edge.src, min_graph_distance)
			dst_distance = vertex_distances.get(edge.dst, min_graph_distance)
			if src_distance + closest_point.distance(edge.src.point) >= min_graph_distance and dst_distance + closest_point.distance(edge.dst.point) >= min_graph_distance:
				dst_edge = edge
				dst_point = closest_point
				break

		for offset in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
			adj_point = closest_point.add(geom.Point(offset[0], offset[1]))
			if r.contains(adj_point) and adj_point not in seen_points and src.distance(adj_point) < opp.distance(adj_point):
				distance = closest_distance + 1 + (1 - im[adj_point.x, adj_point.y])
				if adj_point not in distances or distance < distances[adj_point]:
					distances[adj_point] = distance
					prev[adj_point] = closest_point

	if dst_edge is None:
		return None, None

	path = []
	point = dst_point
	while point != src:
		path.append(point)
		point = prev[point]
	path.append(src)
	path.reverse()

	return dst_edge, path


##########################
def write_graph_to_file_original(graph_data, filename="graph_output_original.txt"):
    with open(filename, "w") as file:
        # Tulis jumlah vertex dan edge
        num_vertices = len(graph_data)
        num_edges = sum(len(neighbors) for neighbors in graph_data.values()) // 2
        file.write(f"Jumlah Vertex: {num_vertices}\n")
        file.write(f"Jumlah Edge: {num_edges}\n\n")
        
        # Tulis semua vertex
        file.write("Vertices:\n")
        for vertex in graph_data.keys():
            file.write(f"{vertex}\n")
        
        # Tulis semua edge
        file.write("\nEdges:\n")
        edges = set()  # Untuk menghindari duplikasi karena graph tidak berarah
        for vertex, neighbors in graph_data.items():
            for neighbor in neighbors:
                edge = tuple(sorted([vertex, neighbor]))  # Pastikan tidak ada duplikat
                if edge not in edges:
                    edges.add(edge)
                    file.write(f"{edge}\n")
    
    print(f"Graph telah ditulis ke {filename}")

def write_graph_to_file_bidirectional(graph_data, output_path):
    # Kumpulkan semua vertex unik
    vertex_set = set(graph_data.keys())
    for neighbors in graph_data.values():
        vertex_set.update(neighbors)

    # Buat list dan mapping index
    vertices = list(vertex_set)
    vertex_index = {v: i for i, v in enumerate(vertices)}

    # Buat edges dalam format index (bidirectional)
    edges = []
    for source, targets in graph_data.items():
        for target in targets:
            idx1 = vertex_index[source]
            idx2 = vertex_index[target]
            edges.append((idx1, idx2))
            edges.append((idx2, idx1))  # dua arah

    with open(output_path, 'w') as f:
        for v in vertices:
            f.write(f"{v[0]} {v[1]}\n")
        
        f.write("\n")
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")

def parse_graph(file_path):
     # Open the file
     with open(file_path, 'r') as file:
         # Read all lines from the file
         lines = file.readlines()
     
     # Split the data into two parts: vertices and edges
     vertices = []
     edges = []
     is_edge_section = False
 
     for line in lines:
         line = line.strip()
         if not line:
             is_edge_section = True  # This marks the transition from vertices to edges
             continue
         if is_edge_section:
             # Edge section (index1 index2)
             edges.append(tuple(map(int, line.split())))
         else:
             # Vertex section (x y)
             x, y = map(int, line.split())
             vertices.append((x, y))
     
     # Create the graph structure
     graph = {}
     
     # Create a dictionary to map vertex index to actual vertex
     vertex_index = {i: vertices[i] for i in range(len(vertices))}
     
     # Add edges to the graph
     for source_index, dest_index in edges:
         # Get the vertex pairs for source and destination
         source_vertex = vertex_index[source_index]
         dest_vertex = vertex_index[dest_index]
         
         # Add the edge in both directions (undirected graph)
         if source_vertex not in graph:
             graph[source_vertex] = []
         if dest_vertex not in graph:
             graph[dest_vertex] = []
         
         graph[source_vertex].append(dest_vertex)
         graph[dest_vertex].append(source_vertex)
 
     return graph
 
def visualize_graph(file_path):
    # Parse the graph data from the file
    graph_data = parse_graph(file_path)

    # Buat graph kosong
    G = nx.Graph()

    # Tambahkan edge dari dictionary
    for node, neighbors in graph_data.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    # Plot graph
    plt.figure(figsize=(10, 10))
    nx.draw(G, pos={node: node for node in G.nodes()}, with_labels=True, node_size=100, font_size=8)
    plt.show()

def save(g, fname):
    coord_to_new_id = {}
    unique_vertices = []
    old_id_to_new_id = {}

    for vertex in g.vertices:
        coord = (vertex.point.x, vertex.point.y)
        if coord not in coord_to_new_id:
            new_id = len(unique_vertices)
            coord_to_new_id[coord] = new_id
            unique_vertices.append(coord)
        old_id_to_new_id[vertex.id] = coord_to_new_id[coord]

    cleaned_edges = set()
    for edge in g.edges:
        new_src = old_id_to_new_id[edge.src.id]
        new_dst = old_id_to_new_id[edge.dst.id]
        if new_src != new_dst:
            cleaned_edges.add(tuple(sorted((new_src, new_dst))))

    with open(fname, 'w') as f:
        for x, y in unique_vertices:
            f.write(f"{x} {y}\n")
        f.write("\n")
        for src, dst in sorted(cleaned_edges):
            f.write(f"{src} {dst}\n")

def restruct_graph(input_txt_path, output_pickle_path):
    """
    Mengubah file teks berisi vertex dan edge (dipisahkan oleh baris kosong)
    menjadi graph adjacency list berbasis tuple koordinat, dan simpan sebagai file .p
    """
    with open(input_txt_path, 'r') as f:
        raw_lines = f.readlines()

    # Pisahkan antara vertex dan edge berdasarkan baris kosong
    split_index = raw_lines.index('\n') if '\n' in raw_lines else len(raw_lines)
    vertex_lines = [line.strip() for line in raw_lines[:split_index] if line.strip()]
    edge_lines = [line.strip() for line in raw_lines[split_index + 1:] if line.strip()]

    # Parsing vertex
    vertices = [tuple(map(int, line.split())) for line in vertex_lines]

    # Inisialisasi graph kosong
    graph = {v: [] for v in vertices}

    # Parsing edge
    for line in edge_lines:
        i, j = map(int, line.split())
        if i < len(vertices) and j < len(vertices):
            v1 = vertices[i]
            v2 = vertices[j]
            graph[v1].append(v2)
            graph[v2].append(v1)  # Bidirectional

    # Simpan ke file .p
    with open(output_pickle_path, 'wb') as out_file:
        pickle.dump(graph, out_file)

######################################

# Main
if __name__ == "__main__":
    outim = imageio.imread(r"D:\Kuliah\bismillah-yudis-1\Tools\graph\inferencer_spacenet\mask\AOI_2_Vegas_1173_road.png").astype('float32') / 255.0
    outim = outim.swapaxes(0, 1)

    with open(r'D:\Kuliah\bismillah-yudis-1\Tools\graph\inferencer_spacenet\graph\AOI_2_Vegas_1173.p', 'rb') as file:
        graph_data = pickle.load(file)

    # write_graph_to_file_original(graph_data)
    write_graph_to_file_bidirectional(graph_data, 'save.txt')
    
    g = graph.read_graph(r"D:\Kuliah\bismillah-yudis-1\Tools\graph\save.txt")
    connections = get_connections(g, outim)
    g = insert_connections(g, connections)
    save(g, 'save.txt')

    restruct_graph('save.txt', 'save.p')

    # Jika ingin visualisasi save.txt
    visualize_graph('save.txt')

    # Jika ingin visualisasi save.p
    with open('save.p', 'rb') as file:
        graph_data = pickle.load(file)
    
    G = nx.Graph()
    for node, neighbors in graph_data.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    plt.figure(figsize=(10, 10))
    nx.draw(G, pos={node: node for node in G.nodes()}, with_labels=True, node_size=100, font_size=10)
    plt.show()
