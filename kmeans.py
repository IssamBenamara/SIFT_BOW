from sklearn import metrics 
from sklearn.cluster import KMeans

def compute_visual_dict(sift, n_clusters=1000, n_init=1, verbose=1):
    # reorder data
    dim_sift = sift[0].shape[-1]
    sift = [s.reshape(-1, dim_sift) for s in sift]
    sift = np.concatenate(sift, axis=0)
    # remove zero vectors
    keep = ~np.all(sift==0, axis=1)
    sift = sift[keep]
    # randomly pick sift
    ids, _ = compute_split(sift.shape[0], pc=0.05)
    sift = sift[ids]
    sift = sift/255

    kmeans = KMeans(n_clusters=n_clusters, n_init = n_init).fit(sift)
    cluster_centers = kmeans.cluster_centers_.tolist()
    cluster_centers.append([0]*(len(cluster_centers[0])))
    return np.asarray(cluster_centers)
