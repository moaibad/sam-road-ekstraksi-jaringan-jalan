import pickle
from PIL import Image
import numpy as np
import networkx as nx
from skimage import measure
from shapely.geometry import LineString

# 6. Muat kembali graph dari file pickle
with open("D:\Kuliah\\bismillah-yudis-1\Tools\sam_road\spacenet\RGB_1.0_meter\AOI_5_Khartoum_471__gt_graph.p", "rb") as f:
    loaded_graph = pickle.load(f)

print(f"Graph berhasil dimuat kembali: {loaded_graph}")