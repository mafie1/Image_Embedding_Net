def _cluster_points(self, points):
    cluster_ids = []
    cluster_idx = 0
    cluster_centers = []

    for i, point in enumerate(points):
        if (len(cluster_ids) == 0):
            cluster_ids.append(cluster_idx)
            cluster_centers.append(point)
            cluster_idx += 1
        else:
            for center in cluster_centers:
                dist = distance(point, center)
                if (dist < CLUSTER_THRESHOLD):
                    cluster_ids.append(cluster_centers.index(center))
            if (len(cluster_ids) < i + 1):
                cluster_ids.append(cluster_idx)
                cluster_centers.append(point)
                cluster_idx += 1
    return cluster_ids