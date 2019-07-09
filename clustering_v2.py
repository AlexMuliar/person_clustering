from db_config import connection
from PIL import Image
from sklearn.cluster import DBSCAN
from scipy.optimize import minimize


# from cuml import DBSCAN
# import cudf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os, sys

save_to = 'test_folder'
try:
    os.system(f'rm -rf { save_to }')
except Exception as ex:
    print(ex)
finally:
    os.mkdir(save_to)

size = 16, 16
min_samples = 100
step = 10

n_iter = 1

max_rows = 5000

def get_embedings():
    df = pd.read_sql(f"\
        SELECT id, dress\
        FROM features_raw\
        LIMIT { max_rows }", connection)
    for i in range(df.shape[0]):
        try:
            df.dress[i] = eval(df.iloc[i].dress)[0]
        except Exception:
            df.drop([i])
    df.columns = ['id', 'image']
    print(df.columns)
    return df


def get_images():
    df = pd.read_sql(f"\
        SELECT id, image, width, height\
        FROM object_image\
        WHERE height > 100\
        LIMIT { max_rows }", connection)
    for i in range(df.shape[0]):
        obj = df.iloc[i]
        df.image[i] = Image.frombytes("RGB", (obj.width, obj.height), obj.image).convert('L').resize(size)  
        df.image[i] = np.array(df.image[i]).reshape((size[0] * size[1]))
        # if len(df.image[i]) != 256:
        #     print(df.id[i], len(df.image[i]))

    df = df.drop(['width', 'height'], axis='columns')
    #print(df.image.shape)    
    # df = cudf.from_pandas(df)
    return df


def search_distance(data):
    def clustering(eps):
        clf_ = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        clf_.fit(df)
        unique_, counts_ = np.unique(clf_.labels_, return_counts=True)
        counts_ = [i for i in counts_]
        if counts_[1:]:
            return counts_[0] + counts_[counts_.index(max(counts_[1:]))]
        else:
            return counts_[0]

    distance = 1
    plot = dict()    
    df = list(data.image)
    
    while True:
        clf = DBSCAN(eps=distance, min_samples=min_samples, n_jobs=-1)
        clf.fit(df)
        unique, counts = np.unique(clf.labels_, return_counts=True)
        if not -1 in unique:
            break
        counts = [i for i in counts]
        if counts[1:]:
            noise_number = counts[0] + counts[counts.index(max(counts[1:]))]
        else:
            noise_number = counts[0]
        plot.update({distance: noise_number})
        distance += step
        
    key_list = list(plot.keys())
    val_list = list(plot.values())
    minimum = key_list[val_list.index(min(val_list))]

    fig, ax = plt.subplots( nrows=1, ncols=1 )  # create figure & 1 axis
    ax.plot(key_list, val_list)
    fig.savefig(f'./{ minimum }.png')   # save the figure to file
    plt.close(fig)

    print(f"x0: { minimum }")
    res = minimize(clustering, minimum, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    print(f"x: { res.x[0] }")
    return res.x[0]


def marker_clusters(data, eps):
    df = list(data.image)
    clf = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(df)
    unique, counts = np.unique(clf.labels_, return_counts=True)
    if len(unique) == 1:
        return -1, data
    counts = [i for i in counts]
    noise_index = counts.index(max(counts[1:])) - 1
    for i in range(len(clf.labels_)):
        if clf.labels_[i] == noise_index:
            clf.labels_[i] = -1
    data['cluster'] = clf.labels_
    return 1, data

def iterator(n):
    global n_iter
    global min_samples
    global step

    if len(sys.argv) > 1 and sys.argv[1] == 'image':
        data = get_images()
    else:
        data = get_embedings()
    
    max_cluster = 0
    print(data.image.shape)
    for _ in range(n):
        print(f'Photos: { data.shape[0] }')
        dist = search_distance(data)
        label, data = marker_clusters(data, dist)
        print(f'Finded: { len(data.cluster.unique()) - 1 }')
        if label == -1:
            break
        for i in data.cluster.unique():
            if i == -1:
                continue
            os.mkdir(f'{ save_to }/{ i + max_cluster }')
        for i in range(data.shape[0]):
            if data.iloc[i].cluster == -1:
                continue
            item = pd.read_sql(f"\
                SELECT id, image, width, height\
                FROM object_image\
                WHERE id = { data.iloc[i].id }", connection).iloc[0]
            item.image = Image.frombytes("RGB", (item.width, item.height), item.image)
            item.image.save(f'{ save_to }/{ data.iloc[i].cluster + max_cluster }/{ item.id }.jpg')
        max_cluster += max(data.cluster.unique()) + 1
        data = data[data.cluster == -1]
        n_iter += 1
        min_samples -= int(min_samples / 4)
        step = min_samples / 4
    os.mkdir(f'{ save_to }/-1')
    for i in range(data.shape[0]):
        item = pd.read_sql(f"\
                SELECT id, image, width, height\
                FROM object_image\
                WHERE id = { data.iloc[i].id }", connection).iloc[0]
        item.image = Image.frombytes("RGB", (item.width, item.height), item.image)
        item.image.save(f'{ save_to }/-1/{ item.id }.jpg')


if __name__ == "__main__":
    import datetime
    start = datetime.datetime.now()
    #os.system('rm -rf result; mkdir result;')
    iterator(20)
    print(datetime.datetime.now() - start)
