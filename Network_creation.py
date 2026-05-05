from pathlib import Path
import warnings
import pandas as pd
import networkx as nx

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset" / "cleaned"
OUTPUTS_DIR = DATA_DIR / "outputs"

AIRPORTS_FILENAME = "airports_cleaned.csv"
ROUTES_FILENAME = "routes_cleaned.csv"

def load_airports(data_dir=DATA_DIR, filename=AIRPORTS_FILENAME):
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File airports non trovato: {path}. "
            "Controllare che il percorso e il nome del file siano corretti."
        )

    airports_df = pd.read_csv(path)

    expected_cols = {
        "airport_id",
        "name",
        "city",
        "country",
        "iata",
        "icao",
        "latitude",
        "longitude",
        "continent"
    }

    missing = expected_cols - set(airports_df.columns)
    if missing:
        warnings.warn(
            f"Mancano alcune colonne attese nel dataset aeroporti: {missing}"
        )

    return airports_df

def load_routes(data_dir=DATA_DIR, filename=ROUTES_FILENAME):
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(
            f"File routes non trovato: {path}. "
            "Controllare che il percorso e il nome del file siano corretti."
        )

    routes_df = pd.read_csv(path)

    expected_cols = {
        "airline",
        "airline_id",
        "source_airport_id",
        "destination_airport_id",
        "stops"
    }

    missing = expected_cols - set(routes_df.columns)
    if missing:
        warnings.warn(
            f"Mancano alcune colonne attese nel dataset rotte: {missing}"
        )

    return routes_df

# Controlli preliminari

def filter_valid_routes(routes_df, airports_df):
    clean_routes = routes_df.dropna(
        subset=["source_airport_id", "destination_airport_id"]
    ).copy()

    clean_routes["source_airport_id"] = clean_routes["source_airport_id"].astype("Int64")
    clean_routes["destination_airport_id"] = clean_routes["destination_airport_id"].astype("Int64")

    airports_clean = airports_df.copy()
    airports_clean["airport_id"] = airports_clean["airport_id"].astype("Int64")

    valid_airport_ids = set(airports_clean["airport_id"].dropna().unique())

    before = len(clean_routes)

    clean_routes = clean_routes[
        clean_routes["source_airport_id"].isin(valid_airport_ids)
        & clean_routes["destination_airport_id"].isin(valid_airport_ids)
    ]

    after = len(clean_routes)
    removed = before - after

    if removed > 0:
        warnings.warn(
            f"Sono state rimosse {removed} rotte perché collegate ad aeroporti "
            "non presenti nel dataset degli aeroporti."
        )

    return clean_routes

# Creazione edge list con pesi

def create_edge_list(routes_df):
    required_cols = {"source_airport_id", "destination_airport_id"}
    missing = required_cols - set(routes_df.columns)
    if missing:
        raise ValueError(
            f"Per creare l'edge list mancano queste colonne: {missing}"
        )

    edge_df = (
        routes_df
        .groupby(["source_airport_id", "destination_airport_id"], as_index=False)
        .size()
        .rename(columns={"size": "weight"})
    )

    return edge_df

# Creazione grafo diretto e pesato

def build_airport_graph(edge_df):
    required_cols = {"source_airport_id", "destination_airport_id", "weight"}
    missing = required_cols - set(edge_df.columns)
    if missing:
        raise ValueError(
            f"Per costruire il grafo mancano queste colonne nella edge list: {missing}"
        )

    G = nx.from_pandas_edgelist(
        edge_df,
        source="source_airport_id",
        target="destination_airport_id",
        edge_attr="weight",
        create_using=nx.DiGraph()
    )

    return G

# Aggiunta degli attributi degli aeroporti ai nodi del grafo

def add_airport_attributes(G, airports_df):
    if "airport_id" not in airports_df.columns:
        raise ValueError(
            "La colonna 'airport_id' non è presente nel dataset degli aeroporti. "
            "Impossibile mappare gli attributi sui nodi del grafo."
        )

    airports_clean = airports_df.copy()
    airports_clean["airport_id"] = airports_clean["airport_id"].astype("Int64")

    attr_cols = [col for col in airports_clean.columns if col != "airport_id"]

    attr_data = {}

    for _, row in airports_clean.iterrows():
        airport_id = row["airport_id"]
        if airport_id in G:
            attr_data[airport_id] = {col: row[col] for col in attr_cols}

    nx.set_node_attributes(G, attr_data)

# Calcolo delle misure di flusso base
def compute_basic_flow_metrics(G):
    nodes = list(G.nodes())

    in_degree = {}
    out_degree = {}
    in_strength = {}
    out_strength = {}
    out_in_ratio = {}

    for node in nodes:
        in_d = G.in_degree(node)
        out_d = G.out_degree(node)

        in_s = G.in_degree(node, weight="weight")
        out_s = G.out_degree(node, weight="weight")

        in_degree[node] = in_d
        out_degree[node] = out_d
        in_strength[node] = in_s
        out_strength[node] = out_s

        if in_d > 0:
            out_in_ratio[node] = float(out_d) / float(in_d)
        else:
            out_in_ratio[node] = float("nan")

    metrics_df = pd.DataFrame({
        "node": nodes,
        "in_degree": pd.Series(in_degree),
        "out_degree": pd.Series(out_degree),
        "in_strength": pd.Series(in_strength),
        "out_strength": pd.Series(out_strength),
        "out_in_ratio": pd.Series(out_in_ratio)
    })

    metrics_df.set_index("node", inplace=True)

    return metrics_df

# Metriche di centralità avanzata

def compute_advanced_centrality_metrics(G):

    pagerank = nx.pagerank(G, weight="weight", max_iter=300, tol=1.0e-06)
    hits_hub, hits_auth = nx.hits(G, max_iter=500, normalized=True)
    eigenvector = nx.eigenvector_centrality(G.to_undirected(), max_iter=500, weight="weight")
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    df = pd.DataFrame({
        "node": list(G.nodes()),
        "pagerank": pd.Series(pagerank),
        "hits_hub": pd.Series(hits_hub),
        "hits_authority": pd.Series(hits_auth),
        "eigenvector": pd.Series(eigenvector),
        "betweenness": pd.Series(betweenness)
    })

    df.set_index("node", inplace=True)
    return df

# tabella finale delle metriche + attributi

def build_node_metrics_table(G, basic_metrics_df, advanced_metrics_df, airports_df):
    metrics_df = basic_metrics_df.join(advanced_metrics_df, how="inner")

    airports_clean = airports_df.copy()
    airports_clean["airport_id"] = airports_clean["airport_id"].astype("Int64")
    airports_clean = airports_clean.set_index("airport_id")

    airports_clean = airports_clean.loc[
        airports_clean.index.intersection(list(G.nodes()))
    ]

    metrics_with_attrs_df = metrics_df.join(
        airports_clean,
        how="left"
    )

    return metrics_with_attrs_df

# Aggregazione per continente

def create_continent_flow_table(edge_df, airports_df):
    required_edge_cols = {"source_airport_id", "destination_airport_id", "weight"}
    missing_edge = required_edge_cols - set(edge_df.columns)
    if missing_edge:
        raise ValueError(
            f"Mancano colonne nella edge list per l'aggregazione per continente: {missing_edge}"
        )

    if "airport_id" not in airports_df.columns or "continent" not in airports_df.columns:
        raise ValueError(
            "airports_df deve contenere le colonne 'airport_id' e 'continent' "
            "per poter aggregare per continente."
        )

    edge_clean = edge_df.copy()
    edge_clean["source_airport_id"] = edge_clean["source_airport_id"].astype("Int64")
    edge_clean["destination_airport_id"] = edge_clean["destination_airport_id"].astype("Int64")

    airports_small = airports_df[["airport_id", "continent"]].copy()
    airports_small["airport_id"] = airports_small["airport_id"].astype("Int64")

    airports_source = airports_small.rename(
        columns={
            "airport_id": "source_airport_id",
            "continent": "source_continent"
        }
    )
    merged = edge_clean.merge(
        airports_source,
        on="source_airport_id",
        how="left"
    )

    airports_target = airports_small.rename(
        columns={
            "airport_id": "destination_airport_id",
            "continent": "target_continent"
        }
    )
    merged = merged.merge(
        airports_target,
        on="destination_airport_id",
        how="left"
    )

    before = len(merged)
    merged = merged.dropna(subset=["source_continent", "target_continent"])
    after = len(merged)
    removed = before - after

    if removed > 0:
        warnings.warn(
            f"Sono state rimosse {removed} tratte perché prive di informazione sul continente."
        )

    continent_flows = (
        merged
        .groupby(["source_continent", "target_continent"], as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "flow"})
    )

    return continent_flows

# Costruzione grafo tra continenti

def build_continent_graph(continent_flows_df):
    required_cols = {"source_continent", "target_continent", "flow"}
    missing = required_cols - set(continent_flows_df.columns)
    if missing:
        raise ValueError(
            f"Per costruire il grafo dei continenti mancano queste colonne: {missing}"
        )

    Gc = nx.DiGraph()

    for _, row in continent_flows_df.iterrows():
        src = row["source_continent"]
        tgt = row["target_continent"]
        flow = row["flow"]

        if pd.isna(src) or pd.isna(tgt):
            continue

        if Gc.has_edge(src, tgt):
            Gc[src][tgt]["flow"] += flow
        else:
            Gc.add_edge(src, tgt, flow=flow)

    return Gc

def compute_continent_directional_metrics(continent_flows_df):
    required_cols = {"source_continent", "target_continent", "flow"}
    missing = required_cols - set(continent_flows_df.columns)
    if missing:
        raise ValueError(
            f"Per calcolare le metriche direzionali sui continenti mancano queste colonne: {missing}"
        )

    out_flow_series = (
        continent_flows_df
        .groupby("source_continent")["flow"]
        .sum()
        .rename("out_flow")
    )

    in_flow_series = (
        continent_flows_df
        .groupby("target_continent")["flow"]
        .sum()
        .rename("in_flow")
    )

    continents = sorted(
        set(out_flow_series.index).union(set(in_flow_series.index))
    )
    directional_df = pd.DataFrame(index=continents)

    directional_df = directional_df.join(out_flow_series, how="left")
    directional_df = directional_df.join(in_flow_series, how="left")

    directional_df["out_flow"] = directional_df["out_flow"].fillna(0)
    directional_df["in_flow"] = directional_df["in_flow"].fillna(0)

    directional_df["net_flow"] = directional_df["out_flow"] - directional_df["in_flow"]

    directional_df["out_in_ratio"] = directional_df["out_flow"] / directional_df["in_flow"]
    directional_df.loc[directional_df["in_flow"] == 0, "out_in_ratio"] = float("nan")

    directional_df.index.name = "continent"
    directional_df.reset_index(inplace=True)

    return directional_df

def compute_airport_communities(G):
    G_und = nx.Graph()

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)

        if G_und.has_edge(u, v):
            G_und[u][v]["weight"] += w
        else:
            G_und.add_edge(u, v, weight=w)

    communities = nx.algorithms.community.greedy_modularity_communities(G_und, weight="weight")

    rows = []
    for cid, comm in enumerate(communities):
        for node in comm:
            rows.append({"node": node, "community_id": cid})

    community_df = pd.DataFrame(rows)

    return community_df

# Salvataggio

def save_node_metrics(node_metrics_df, output_dir=OUTPUTS_DIR, filename="node_metrics.csv"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    node_metrics_df.to_csv(out_path, index=True, index_label="airport_id")

    print(f"File metriche aeroporti salvato in: {out_path}")

def save_continent_flows(continent_flows_df, output_dir=OUTPUTS_DIR, filename="continent_flows.csv"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    continent_flows_df.to_csv(out_path, index=False)

    print(f"File flussi tra continenti salvato in: {out_path}")

def save_continent_directional_metrics(continent_directional_df,
                                       output_dir=OUTPUTS_DIR,
                                       filename="continent_directional_metrics.csv"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    continent_directional_df.to_csv(out_path, index=False)

    print(f"File metriche direzionali per continente salvato in: {out_path}")

def save_airport_communities(community_df,
                             output_dir=OUTPUTS_DIR,
                             filename="airport_communities.csv"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    to_save = community_df.rename(columns={"node": "airport_id"})

    to_save.to_csv(out_path, index=False)

    print(f"File comunità aeroporti salvato in: {out_path}")


if __name__ == "__main__":
    # STEP A2: carico solo gli aeroporti (le rotte arriveranno nello step successivo)
    airports_df = load_airports()
    routes_df = load_routes()

    print(f"Airports loaded: {len(airports_df)} righe")
    print(f"Routes loaded: {len(routes_df)} righe")

    # STEP A3: controlli preliminari sulle rotte
    routes_filtered_df = filter_valid_routes(routes_df, airports_df)
    print(f"Routes after filtering: {len(routes_filtered_df)} righe")

    # STEP A4: creazione edge list aggregata
    edge_df = create_edge_list(routes_filtered_df)
    print(f"Edge list size: {len(edge_df)} archi")

    # STEP A5: costruzione del grafo diretto e pesato
    G = build_airport_graph(edge_df)
    print(f"Grafo: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

    # STEP A6: aggiunta attributi degli aeroporti ai nodi
    add_airport_attributes(G, airports_df)

    # STEP B1: calcolo delle misure di flusso base
    basic_metrics_df = compute_basic_flow_metrics(G)

    print("Prime righe delle metriche di flusso base:")
    print(basic_metrics_df.head())

    # STEP B2: metriche di centralità avanzata
    advanced_metrics_df = compute_advanced_centrality_metrics(G)

    print("Centralità avanzate calcolate.")
    print(advanced_metrics_df.head())

    # STEP B3: tabella finale delle metriche + attributi
    node_metrics_df = build_node_metrics_table(
        G,
        basic_metrics_df,
        advanced_metrics_df,
        airports_df
    )

    print(f"Grafo: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
    print("Prime righe della tabella finale delle metriche:")
    print(node_metrics_df.head())

    # STEP C1: aggregazione dei flussi per continente
    continent_flows_df = create_continent_flow_table(edge_df, airports_df)
    print("Prime righe dei flussi tra continenti: ")
    print(continent_flows_df.head())

    # STEP C2: costruzione del grafo tra continenti
    G_continents = build_continent_graph(continent_flows_df)
    print(f"\nGrafo continenti: {G_continents.number_of_nodes()} nodi, {G_continents.number_of_edges()} archi")

    # STEP C3: metriche direzionali aggregate per continente
    continent_directional_df = compute_continent_directional_metrics(continent_flows_df)
    print("\nMetriche direzionali per continente:")
    print(continent_directional_df)

    # STEP C4: community detection
    community_df = compute_airport_communities(G)

    print("\nEsempio comunità aeroporti:")
    print(community_df.head())
    print(f"Numero comunità trovate: {community_df['community_id'].nunique()}")

    # STEP D1: salvataggio metriche per aeroporto
    save_node_metrics(node_metrics_df)

    # STEP D2: salvataggio flussi tra continenti
    save_continent_flows(continent_flows_df)

    # STEP D3: salvataggio metriche direzionali per continente
    save_continent_directional_metrics(continent_directional_df)

    # STEP D4: salvataggio comunità aeroporti
    save_airport_communities(community_df)