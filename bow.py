# Compute BoW for an image

def compute_feats(vdict, image_sifts):
    from scipy.spatial import distance_matrix
    # TODO compute BoW from `image_sifts`
    # Coding
    dm = distance_matrix(image_sifts,vidct)
    ids = np.argmin(dm,axis=1)
    h = np.zeros(dm.shape)
    h[range(len(ids)),ids] = 1

    # Pooling
    z = h.sum(axis=0)
    z = z/np.linalg.norm(z)
    return z