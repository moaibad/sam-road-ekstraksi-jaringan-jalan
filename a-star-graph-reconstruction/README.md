# 🛣️ A-Star Graph Reconstruction

This repository contains the implementation of the **A\*** (A-Star) algorithm for **road topology reconstruction**, developed as part of my **undergraduate final project**.

In many real-world geospatial applications, road extraction from satellite imagery or semantic segmentation often results in **incomplete road networks** due to **occlusion** — such as buildings, trees, or artifacts in the data. These broken connections can hinder downstream tasks like routing, navigation, and urban planning.

This project proposes a graph-based solution using the A\* search algorithm to automatically **reconnect** the missing segments by:
- Identifying disconnected nodes or endpoints
- Estimating the most likely connections using cost + heuristic
- Reconstructing the missing paths to form a complete topology
