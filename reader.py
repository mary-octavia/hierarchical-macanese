# -*- coding: utf-8 -*-
import re
import codecs
import string
import random
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from time import time
from sklearn import metrics
from scipy.sparse import vstack
from itertools import cycle
from sklearn.cluster import KMeans, AffinityPropagation, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, RandomizedPCA, NMF
from sklearn.preprocessing import scale, Normalizer, Binarizer 
from sklearn.datasets.samples_generator import make_swiss_roll

br_tr = "Train/br.txt"
mo_tr = "Train/mo.txt"
pt_tr = "Train/pt.txt"

br_ts = "Dev/br.txt"
mo_ts = "Dev/mo.txt"
pt_ts = "Dev/pt.txt"

def get_preprocessor():
    def preprocess(unicode_text):
    	replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))
    	unicode_text = ((unicode_text.encode("utf8")).translate(replace_punctuation)).decode("utf8")
    	unicode_text = ((unicode_text.encode("utf8")).translate(None, "1234567890")).decode("utf8")
        return unicode(unicode_text)
    return preprocess

def extract_features(words, n, count=True, reduced=True, n_labels=3):
	# vectorizer = CountVectorizer(analyzer='char', ngram_range=(n, n), preprocessor=get_preprocessor())
	vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, n), binary=count, preprocessor=get_preprocessor())
	# vectorizer = TfidfVectorizer(analyzer='word', max_df=0.5, max_features=300, min_df=2,use_idf=True, preprocessor=get_preprocessor())
	# transformed_words = vectorizer.fit_transform(words).toarray()
	# transformed_words = np.array(transformed_words, dtype=np.float)
	transformed_words = vectorizer.fit_transform(words)	
	
	# br = vectorizer.fit_transform(words[0:18000])
	# mo = vectorizer.fit_transform(words[18000:36000])
	# pt = vectorizer.fit_transform(words[36000:])
	# transformed_words = vstack([br,mo,pt])

	if reduced:
		svd = TruncatedSVD(n_labels)
		normalizer = Normalizer(copy=False)
		lsa = make_pipeline(svd, normalizer)
		reduced_X = lsa.fit_transform(transformed_words)
		return reduced_X, svd
	else:
		return transformed_words


def load_data(fin1=br_ts, fin2=mo_ts, fin3=pt_ts, labels=True):
	''' read articles from files
	'''
	# br, mo, pt = [], [], []

	all_art, y = [], []
	with codecs.open(fin1, "r",encoding="utf-8") as f:
		for line in f:
			line = line.replace("\n", "")
			# br.append(line)
			all_art.append(line)
			y.append(1) #1 for brazilian articles

	with codecs.open(fin2, "r",encoding="utf-8") as f:
		for line in f:
			line = line.replace("\n", "")
			# mo.append(line)
			all_art.append(line)
			y.append(2) #2for mocanese articles

	with codecs.open(fin3, "r", encoding="utf-8") as f:
		for line in f:
			line = line.replace("\n", "")
			# pt.append(line)
			all_art.append(line)
			y.append(3) #3for portuguese articles

	with codecs.open("all-articles-raw-ts.txt","w", encoding="utf-8") as f:
		f.write("\n".join(all_art))

	print "total articles: ", len(all_art)
	if labels:
		return all_art, y
	else:
		return all_art


def affinity(articles, labels):
    print "Extracting features..."
    X = extract_features(articles, 3, False)
    X_norms = np.sum(X * X, axis=1)
    S = -X_norms[:, np.newaxis] - X_norms[np.newaxis, :] + 2 * np.dot(X, X.T)
    p = 10 * np.median(S)

    print "Fitting affinity propagation clustering with unknown no of clusters..."
    af = AffinityPropagation().fit(S, p)
    indices = af.cluster_centers_indices_
    for i, idx in enumerate(indices):
        print i, articles[idx].encode("utf8")

    n_clusters_ = len(indices)

    print "Fitting PCA..."
    X = RandomizedPCA(2).fit(X).transform(X)    
    
    print "Plotting..."
    pl.figure(1)
    pl.clf()
    
    colors = cycle('bgrcmyk')
    for k, col in zip(range(n_clusters_), colors):
        class_members = af.labels_ == k
        cluster_center = X[indices[k]]
        pl.plot(X[class_members,0], X[class_members,1], col+'.')
        pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                                         markeredgecolor='k', markersize=14)
        for x in X[class_members]:
            pl.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col) 

    pl.title('Estimated number of clusters: %d' % n_clusters_)
    # pl.show()
    pl.savefig("affinity_cluster.png")


def print_results_to_file(estimator, name, labels):
	f = codecs.open("kmeanscores.txt", "w", encoding="utf-8")
	f.write('init    inertia    homo   compl  v-means     ARI AMI')
	f.write("\n")
	f.write(name + " ")
	f.write(str(estimator.inertia_) +" ")
	f.write(str(metrics.homogeneity_score(labels, estimator.labels_))+" ")
	f.write(str(metrics.completeness_score(labels, estimator.labels_))+ " ")
	f.write(str(metrics.v_measure_score(labels, estimator.labels_))+ " ")
	f.write(str(metrics.adjusted_rand_score(labels, estimator.labels_))+ " ")
	f.write(str(metrics.adjusted_mutual_info_score(labels,  estimator.labels_))+ " ")
	f.close()

def bench_k_means(estimator, name, data, labels):
	np.random.seed(1000)
	estimator.fit(data)
	print_results_to_file(estimator, name, labels)

def k_clusters(kmeans, nclust, data, docs):
	# data = extract_features(infinitives, 3, False)
	# reduced_data = PCA(n_components=2).fit_transform(data)
	# kmeans = KMeans(n_clusters=nclust, n_init=1).fit(data)
	f = codecs.open("kclusters.txt", "w", encoding="utf-8")

	nn = KNeighborsClassifier(1).fit(data, np.zeros(data.shape[0]))
	_, idx = nn.kneighbors(kmeans.cluster_centers_)
	f.write("\ncentroids:\n")
	for centr in docs[idx.flatten()]:
		f.write(centr+"\n")

	f.write("top 10 docs per cluster:\n")	
	order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
	for i in range(nclust):
		f.write("Cluster " +str(i)+":\n")
		for ind in order_centroids[i, :10]:
			f.write("with centroid: "+ str(order_centroids[i]))
			f.write("\n" +str(ind)+ " "+ docs[ind]+"\n")
	f.close()

def get_random_points(n):
	'''gets random indexes for data points to plot'''
	clen = int(n/3)
	idx1 = np.linspace(0, clen-1, clen, dtype=int)
	idx2 = np.linspace(clen, clen*2-1, clen*2, dtype=int)
	idx3 = np.linspace(clen*2, clen*3-1, clen*3, dtype=int)

	num =  int(clen/1000)        # set the number to select here.
	lstr = random.sample(idx1, num)
	lstr = lstr + random.sample(idx2, num)
	lstr = lstr + random.sample(idx3, num)
	return lstr

def visualize_kclusters(data, n_labels, labels):
	# Visualize the results on PCA-reduced data
	# reduced_data = PCA(n_components=2).fit_transform(data)
	pca = PCA(n_components=2).fit(data)
	reduced_data = pca.transform(data)
	# reduced_data = data #for legacy reasons
	# svd = TruncatedSVD(2)
	# normalizer = Normalizer(copy=False)
	# lsa = make_pipeline(svd, normalizer)
	# lsa = Pipeline([('svd',svd),('normalizer',normalizer)])
	# reduced_data = lsa.fit_transform(data)
	
	kmeans = KMeans(init='k-means++', n_clusters=n_labels, n_init=1)
	kmeans.fit(reduced_data)

	print_results_to_file(kmeans, "k-means++", labels)

	# Step size of the mesh. Decrease to increase the quality of the VQ.
	h = .004     # point in the mesh [x_min, m_max]x[y_min, y_max].

	# Plot the decision boundary. For that, we will assign a color to each
	x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
	y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# print "xx: ", xx, "yy:", yy

	# Obtain labels for each point in mesh. Use last trained model.
	Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

	# Put the result into a color plot
	Z = Z.reshape(xx.shape)
	plt.figure(1)
	plt.clf()
	plt.imshow(Z, interpolation='nearest',
	           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
	           cmap=plt.cm.Pastel1,
	           aspect='auto', origin='lower')

	plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels)
	# Plot the centroids as a blue X
	centroids = kmeans.cluster_centers_
	plt.scatter(centroids[:, 0], centroids[:, 1],
	            marker='x', s=169, linewidths=3,
	            color='w', zorder=10)

	# plt.title('K-means clustering on all the articles (PCA-reduced data)\n'
	#           'Centroids are marked with white cross')
	plt.title('K-means clustering on all the articles (PCA-reduced data)\n'
	          'Centroids are marked with white cross')
	plt.xlim(x_min, x_max)
	plt.ylim(y_min, y_max)
	plt.xticks(())
	plt.yticks(())
	# plt.show()
	# plt.legend()
	plt.savefig("kmeans3.png")
	return kmeans, reduced_data

def plot_projection(model, data, title, ngram=3):
    fig = plt.figure()
    # Binary model: n-gram appears or not
    for i in range(1, ngram):  # n-gram length (1 to 3)    
        plt.subplot(2, 3, i)
        data = extract_features(data, i, False, False)
        projected_data = model.fit(data).transform(data)
        plt.scatter(projected_data[:, 0], projected_data[:, 1])
        plt.title('Binary %d-grams' % i)
    # pl.show()
    plt.savefig("figure_binary-1-"+str(ngram)+"grampca.png")
    # Frequency model: count the occurences
    for i in range(1, ngram):
        plt.subplot(2, 3, 3+i)
        data = extract_features(data, i, True, False)
        projected_data = model.fit(data).transform(data)
        plt.scatter(projected_data[:, 0], projected_data[:, 1])
        plt.title('Count %d-grams' % i)
    fig.text(.5, .95, title, horizontalalignment='center')
    # fig.legend("la", "lala", "lalala")
    print "lala"
    # pl.show()
    plt.savefig("figure_count-1-"+str(ngram)+"grampca.png")

def plot_hierarchical(X, n_labels):
	# Compute clustering
	print("Compute unstructured hierarchical clustering...")
	st = time()
	svd = TruncatedSVD(3)
	normalizer = Normalizer(copy=False)
	lsa = make_pipeline(svd, normalizer)
	X = lsa.fit_transform(X)

	ward = AgglomerativeClustering(n_clusters=n_labels, linkage='ward').fit(X)
	elapsed_time = time() - st
	label = ward.labels_
	print("Elapsed time: %.2fs" % elapsed_time)
	print("Number of points: %i" % label.size)

	# Plot result
	fig = plt.figure()
	ax = p3.Axes3D(fig)
	ax.view_init(7, -80)
	for l in np.unique(label):
	    ax.plot3D(X[label == l, 0], X[label == l, 1], X[label == l, 2],
	              'o', color=plt.cm.jet(np.float(l) / np.max(label + 1)))
	plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)
	plt.savefig("dendogram.png")

if __name__ == '__main__':
	articles, y = load_data(br_ts, mo_ts, pt_ts, True)
	# n_labels = len(np.unique(y))
	# reduced_data, svd =  extract_features(articles,1, False, True, n_labels)
	# data =  extract_features(articles, 1, False, False)

	# bench_k_means(KMeans(init='k-means++', n_clusters=n_labels, n_init=10), name="k-means++", data=data, labels=y)
	# bench_k_means(KMeans(init='random', n_clusters=n_labels, n_init=10), name="random", data=data, labels=y)
	# visualize_kclusters(data, n_labels, y)

	# svd = TruncatedSVD(2)
	# normalizer = Normalizer(copy=False)
	# lsa = make_pipeline(svd, normalizer)
	# lsa = Pipeline([('svd',svd),('normalizer',normalizer)])
	# reduced_data = lsa.fit_transform(data)
	plot_projection(RandomizedPCA(n_components=2), articles, "PCA projection of articles")
	# plot_projection(lsa, articles, "LSA projection of articles", 2)

	# plot_hierarchical(data, n_labels)

	# affinity(articles[:2000], y) 
