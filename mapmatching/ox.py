import os
import pandas as pd
import osmnx
import geopandas as gpd
from shapely.geometry import Point, LineString

basedir = os.path.abspath(os.path.dirname(__file__)) + '/.cache/'
try:
    os.mkdir(basedir)
except FileExistsError:
    pass
print(f"Made {basedir}")


def csv_to_gpd(fname):
    df = pd.read_csv(fname, index_col=0, low_memory=False)
    df["geometry"] = df["geometry"].apply(
        lambda x: list(map(float, x.replace("(", "").replace(")", "").replace(",", "").split(" ")[1:])))
    if df.shape[1] == 6:
        df["geometry"] = df["geometry"].apply(lambda x: Point(x[0], x[1]))
    else:
        df["geometry"] = df["geometry"].apply(lambda x: LineString([(x[i], x[i + 1]) for i in range(0, len(x), 2)]))
    gdf = gpd.GeoDataFrame(df)
    if gdf.shape[1] == 6:
        gdf.gdf_name = 'map_nodes'
    else:
        gdf.gdf_name = 'map_edges'
        gdf["width"] = gdf["width"].astype("object")
        gdf["lanes"] = gdf["lanes"].astype("object")
    return gdf



def get_graph(name, network_type='dirve', latlng_to_meter_scaler=None):
    graph = osmnx.graph_from_place(name, network_type=network_type)
    node_fname = basedir + f'{name.srtip().replace(", ", "_")}_nodes.csv'
    edge_fname = basedir + f'{name.srtip().replace(", ", "_")}_edges.csv'
    if no_cache or (not os.path.exists(node_fname)) or (not os.path.exists(edge_fname)):
        print('Downloading graph data via osmnx api...')
        nodes, edges = osmnx.save_load.graph_to_gdfs(graph,
                        nodes=True, edges=True, node_geometry=True, fill_edge_geometory=True)[:2]
        nodes.to_csv(node_fname)
        edges.to_csv(edge_fname)

    nodes = csv_to_gpd(node_fname)
    edges = csv_to_gpd(edge_fname)
    edges["oneway"] = edges["oneway"].apply(lambda x: False if "False" in str(x) else True)

    osmids = nodes["osmid"].values
    node_lats = nodes["y"].values.reshape(-1, 1)
    node_lngs = nodes["x"].values.reshape(-1, 1)
    if latlng_to_meter_scaler is None:
        s = np.array([[111000.0, 91000.0]]) # latlng -> meter scale. approximated at Tokyo.
    else:
        latlng_to_meter_scaler = np.array(latlng_to_meter_scaler)
        s = latlng_to_meter_scaler.reshape(1, 2)

    node_latlngs = (s * np.concatenate([node_lats, node_lngs], axis=1).copy()).astype(np.long)

    to_idx = dict()
    for i in range(osmids.shape[0]):
        to_idx[osmids[i]] = i

    mask = edges["u"].apply(lambda x: x in to_idx).values & edges["v"].apply(lambda x: x in to_idx)
    edges = edges[mask]
    print(edges.shape)

    edge_polylines = [np.array(x.coords[:], copy=False) for x in edges["geometry"].values]
    edge_polylines = [x[:, ::-1] * s for x in edge_polylines]

    mbb = np.array([[x[:, 0].min(), x[:, 0].max(),
                       x[:, 1].min(), x[:, 1].max()] for x in edge_polylines])



