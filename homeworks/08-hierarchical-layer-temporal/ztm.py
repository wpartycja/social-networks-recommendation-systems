import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re

def parse_stops(file_path):
    stops_df = pd.read_csv(file_path)
    stops_dict = {
        row['stop_id']: {
            'name': row['stop_name'],
            'latitude': row['stop_lat'],
            'longitude': row['stop_lon']
        }
        for _, row in stops_df.iterrows()
    }
    return stops_dict


def normalize_times(time_str):
    """
    Normalize times greater than 24 hours (e.g., "25:00:00" -> "01:00:00").
    """
    match = re.match(r"^(\d+):(\d{2}):(\d{2})$", time_str)
    if match:
        hours, minutes, seconds = map(int, match.groups())
        if hours >= 24:
            hours = hours % 24
        return f"{hours:02}:{minutes:02}:{seconds:02}"
    return time_str


def parse_stop_times(file_path, trip_service_map):
    stop_times_df = pd.read_csv(file_path)

    stop_times_df['departure_time'] = stop_times_df['departure_time'].apply(normalize_times)

    stop_times_df.sort_values(by=['trip_id', 'stop_sequence'], inplace=True)
    grouped = stop_times_df.groupby('trip_id')
    connections = []
    subset_stop_ids = set()
    for trip_id, group in grouped:
        stops = group['stop_id'].tolist()
        subset_stop_ids.update(stops)
        route_id = trip_service_map.get(trip_id)
        connections.extend([(stops[i], stops[i + 1], route_id) for i in range(len(stops) - 1)])
    return connections, subset_stop_ids


def parse_trips(file_path):
    trips_df = pd.read_csv(file_path)
    trip_to_route = {
        row['trip_id']: row['route_id']
        for _, row in trips_df.iterrows()
    }
    return trip_to_route


def parse_routes(file_path):
    routes_df = pd.read_csv(file_path)

    route_type_colors = {
        0: 'red',   # Tram
        2: 'green', # Train
        3: 'skyblue', # Bus
    }

    route_type_map = {
        row.route_id: row.route_type
        for row in routes_df.itertuples()
    }

    route_colors = {
        row.route_id: route_type_colors.get(row.route_type, 'gray')
        for row in routes_df.itertuples()
    }
    
    return route_colors, route_type_map


def build_graph(stops_dict, connections, subset_stop_ids, route_type_map):
    G = nx.DiGraph()
    missing_stops = set()

    # Add nodes with default route_types
    for stop_id in subset_stop_ids:
        if stop_id in stops_dict:
            G.add_node(stop_id, **stops_dict[stop_id], route_types=set())
        else:
            missing_stops.add(stop_id)

    # Add edges and populate route_types for nodes
    for u, v, route_id in connections:
        route_type = route_type_map.get(route_id, None)
        if u not in missing_stops and v not in missing_stops:
            G.add_edge(u, v, route_type=route_type)
            G.nodes[u]['route_types'].add(route_type)
            G.nodes[v]['route_types'].add(route_type)

    # Ensure all nodes have at least a default route_type if empty
    for node in G.nodes:
        if not G.nodes[node]['route_types']:
            G.nodes[node]['route_types'].add(None)  # Or any default value

    if missing_stops:
        print(f"Warning: {len(missing_stops)} stop IDs are missing in stops_dict: {missing_stops}")

    return G


def draw_graph(G, geographic=False):
    pos = None
    if geographic:
        pos = {
            node: (data['longitude'], data['latitude'])
            for node, data in G.nodes(data=True)
            if data.get('longitude') is not None and data.get('latitude') is not None
        }
        if not pos:
            print("Error: No valid positions available for geographic layout.")
            return
    else:
        pos = nx.spring_layout(G)

    plt.figure(figsize=(18, 12))

    route_type_colors = {
        0: 'red',
        2: 'green',
        3: 'skyblue',
        None: 'gray'
    }

    for route_type, color in route_type_colors.items():
        route_edges = [
            (u, v) for u, v, data in G.edges(data=True)
            if data.get('route_type') == route_type
        ]
        route_nodes = [
            node for node, data in G.nodes(data=True)
            if route_type in data.get('route_types', set())
        ]
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color=[color], alpha=0.6, width=0.5)
        nx.draw_networkx_nodes(G, pos, nodelist=route_nodes, node_color=[color], node_size=5)

    plt.title("Warsaw Transit Network (Colored by Route Type)")
    plt.show()


def extract_subgraph_by_route_type(G, route_type):
    filtered_nodes = [node for node, data in G.nodes(data=True) if route_type in data['route_types']]
    subgraph = G.subgraph(filtered_nodes).copy()
    
    edges_to_remove = [
        (u, v) for u, v, data in subgraph.edges(data=True)
        if data.get('route_type') != route_type
    ]
    subgraph.remove_edges_from(edges_to_remove)
    
    return subgraph


def calculate_network_metrics(G, name):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / N
    assortativity = nx.degree_assortativity_coefficient(G)
    

    density = nx.density(G)
    in_degrees = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    avg_in_degree = sum(in_degrees.values()) / N if N > 0 else 0
    avg_out_degree = sum(out_degrees.values()) / N if N > 0 else 0
    
    return {
        "Network's Name": name,
        "N": N,
        "E": E,
        "⟨k⟩ - average degree": round(avg_degree, 2),
        "assortativity": round(assortativity, 2),
        "density": density,
    }
