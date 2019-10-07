# Compute BoW for an image

def compute_feats(vdict, image_sifts):
    from scipy.spatial import distance_matrix
    # TODO compute BoW from `image_sifts`
    sift = image_sifts
    sift = [s.reshape(-1, sift[0].shape[-1]) for s in sift]
    sift = np.concatenate(sift, axis=0)
    
    # Coding
    dm = distance_matrix(sift,vidct)**2
    ids = np.argmin(dm,axis=1)
    h = np.zeros(dm.shape)
    h[range(len(ids)),ids] = 1

    # Pooling
    z = h.sum(axis=0)
    z = z/np.linalg.norm(z)
    return z
