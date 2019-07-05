from db_config import connection
from PIL import Image
from sklearn.cluster import DBSCAN
# from cuml import DBSCAN
# import cudf
import pandas as pd
import numpy as np
import os

save_to = 'result'
os.mkdir(save_to)

size = 16, 16
min_samples = 3


def get_data():
    df = pd.read_sql("\
        SELECT id, image, width, height\
        FROM object_image", connection)
    for i in range(df.shape[0]):
        obj = df.iloc[i]
        df.image[i] = Image.frombytes("RGB", (obj.width, obj.height), obj.image).convert('L').resize(size)

    df = df.drop(['width', 'height'], axis='columns')

    # df = cudf.from_pandas(df)
    return df


def search_distance(data):
    distance = 1
    plot = dict()
    step = 1
    images = list()
    for i in data.image:
        image = np.array(i).reshape((size[0] * size[1]))
        images.append(image)
    while True:
        clf = DBSCAN(eps=distance, min_samples=min_samples, n_jobs=-1)
        clf.fit(images)
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
        #print(noise_number, counts)
    key_list = list(plot.keys())
    val_list = list(plot.values())
    return key_list[val_list.index(min(val_list))]


def marker_clusters(data, eps):
    images = list()
    for i in data.image:
        image = np.array(i).reshape((size[0] * size[1]))
        images.append(image)
    clf = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(images)
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
    data = get_data()
    max_cluster = 0
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
    os.mkdir(f'{ save_to }/-1')
    for i in range(data.shape[0]):
        item = pd.read_sql(f"\
                SELECT id, image, width, height\
                FROM object_image\
                WHERE id = { data.iloc[i].id }", connection).iloc[0]
        item.image = Image.frombytes("RGB", (item.width, item.height), item.image)
        item.image.save(f'{ save_to }/-1/{ item.id }.jpg')


if __name__ == "__main__":
    #os.system('rm -rf result; mkdir result;')
    iterator(3)
