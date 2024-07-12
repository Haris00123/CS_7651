from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder,StandardScaler,KBinsDiscretizer
import warnings
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import accuracy_score, fbeta_score,make_scorer, mutual_info_score, silhouette_score, normalized_mutual_info_score, classification_report, confusion_matrix,f1_score, mean_squared_error, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA,FastICA
from sklearn.random_projection import johnson_lindenstrauss_min_dim,SparseRandomProjection,GaussianRandomProjection
from sklearn.manifold import TSNE,trustworthiness,Isomap

warnings.simplefilter('ignore')

def test_clustering(dataset,dataset_name,algorithm,algorithm_name,actual_labels,n_components=[2,5,7,9]):

    #Converting dataset to 2D so it's plottable
    converter=TSNE(n_components=2)
    dataset_xy=converter.fit_transform(dataset)
    
    #Plotting 2d color coded with labels 
    fig,axes=plt.subplots(1,len(n_components)+1)
    plt.title(f'{dataset_name} dataset clusters using {algorithm_name}',fontsize=18)
    
    for c,ax in enumerate(fig.axes):
        n_comps=n_components[c]
        clstr=algorithm()

        pass


def calculate_wcss(data,labels,centroids):
    wcss=0

    for i,mean in enumerate(centroids):
        
        #Filtering on relevant label
        filtered=data[labels==i]

        #Calculating WCCSS distance
        distance=np.sum((filtered-mean)**2,axis=1)

        #Adding to WCSS
        wcss+=np.sum(distance)
    
    return wcss

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# def categorizer(data,categorical_cols):
#     '''Helper function to categorize datasets'''

#     if categorical_cols:
#         initial_categorized=data[:,categorical_cols]
#         new_data=data[:,categorical_cols].copy()
#     else:
#         categorical_cols=[]
#         new_data = None

#     for i in range(data.shape[-1]):
#         if i in categorical_cols:
#             pass
#         else:
#             #Getting Standards
#             mean_data=data[:,i].mean()
#             mean_std=data[:,i].std()

#             #Categorizing 
#             if mean_std<0.001:
#                 inter=np.round((data[:,i]-mean_data)/(1)).astype(int).reshape(-1,1)
#             else:
#                 #inter=np.round((data[:,i]-mean_data)/(mean_std)).astype(int).reshape(-1,1)
#                 inter=np.round((data[:,i]-mean_data)/(1)).astype(int).reshape(-1,1)
#             inter=np.where(inter>=5,5,inter)
#             inter=np.where(inter<=-5,-5,inter)
            
#             #Encoder
#             encoder=OneHotEncoder()
#             transformed_data=encoder.fit_transform(inter).toarray()

#             #Concatenating
#             if new_data is None:
#                 new_data=transformed_data
#             else:
#                 new_data=np.c_[new_data,transformed_data]
#     scaler=StandardScaler()
#     return scaler.fit_transform(new_data)

def categorizer(data,categorical_cols):
    '''Helper function to categorize datasets'''

    if categorical_cols:
        initial_categorized=data[:,categorical_cols]
        non_categorical=[x for x in range(data.shape[-1]) if x not in categorical_cols]
        to_categorize=data[:,non_categorical].copy()
        compile=True
    else:
        categorical_cols=[]
        initial_categorized=None
        to_categorize = data.copy()
        compile=False

    binner=KBinsDiscretizer(n_bins=10, encode='onehot')
    binner.fit(to_categorize)
    new_data=binner.transform(to_categorize).toarray()

    if compile:
        new_data=np.c_[new_data,initial_categorized]

    scaler=StandardScaler()
    return scaler.fit_transform(new_data)

def create_elbow_plot(x,y,ax):
    """Function to plot the elbow plot"""

    new_x=x[1:]
    new_y=np.abs(np.array(y[1:])-np.array(y[:-1]))/np.abs(np.array(y[:-1]))
    ax.plot(new_x,new_y,linestyle='--',color='red',label='% change')
    ax.axis('off')

def experiment_em_clusters(dataset,dataset_name,labels_og,algorithm_name,cluster_graphs=4,repeats=3):
    #Converting dataset to 2D so it's plottable
    n_list=np.arange(1,20)

    converter=TSNE(n_components=2)
    dataset_2d=converter.fit_transform(dataset)

    #Average of 3 runs 
    silhouette_scores=np.zeros(shape=len(n_list))
    wcss_scores=np.zeros(shape=len(n_list))
    BIC_scores=np.zeros(shape=len(n_list))
    AIC_scores=np.zeros(shape=len(n_list))
    EM_dict={}

    for r in [21*(1+i) for i in range(repeats)]:

        #Will be calculated for each run for plots
        all_labels=[]
        
        #Assigning Clusters
        for i,n in enumerate(n_list):
            #Creating Gaussian
            em=GaussianMixture(n_components=n,random_state=r)

            #Fitted labels dataset
            em.fit(dataset)

            #Predictions
            labels=em.predict(dataset)
            all_labels.append(labels)

            #Calculating Silhouette Score
            if n==1:
                pass
            else:
                silhouette_scores[i]=silhouette_score(dataset,labels)/repeats

            #WCSS Score
            wcss_scores[i]=calculate_wcss(dataset,labels,em.means_)/repeats

            #BIC Scores
            BIC_scores[i]=em.bic(dataset)/repeats

            #AIC
            AIC_scores[i]=em.aic(dataset)/repeats

            #Saving EM models for final Run 
            EM_dict[n]=em

        #Plotting 2d color coded with labels 
        fig,axes=plt.subplots(1,len(n_list[:cluster_graphs])+1)
        fig.set_size_inches(35,5)

        plt.suptitle(f'{dataset_name} dataset clusters using {algorithm_name}',fontsize=18)

        for c,ax in enumerate(fig.axes):

            if c>-1:
                #Predictions
                labels=all_labels[c+1]

                #Plotting
                sc=ax.scatter(dataset_2d[:,0],dataset_2d[:,1],c=labels,cmap='plasma')
                #ax.legend()
                ax.set_ylabel('Y');
                ax.set_xlabel('X');
                ax.set_title('{} : COMPONENTS'.format(n_list[c+1]));
                ax.legend(*sc.legend_elements(), title='clusters')

            else:

                sc=ax.scatter(dataset_2d[:,0],dataset_2d[:,1],c=labels_og,cmap='plasma')
                #ax.legend()
                ax.set_ylabel('Y');
                ax.set_xlabel('X');
                ax.set_title('Dataset');      
                ax.legend(*sc.legend_elements(), title='clusters')

        plt.tight_layout()

    #Plotting Silhouette score
    fig=plt.figure()
    fig.set_size_inches(10,5)

    plt.title('Silhouette_scores & Clusters')
    plt.plot(n_list[1:],silhouette_scores[1:])
    plt.scatter(n_list[1:], silhouette_scores[1:], s=100, c='blue', marker='o', edgecolors='black')
    plt.xticks(n_list[1:])
    plt.xlabel('N_clusters')
    plt.ylabel('Silhouette_scores')

    plt.tight_layout()

    #Plotting WCSS
    fig=plt.figure()
    fig.set_size_inches(10,5)

    plt.title('WCSS & Clusters')
    plt.plot(n_list,wcss_scores)
    plt.scatter(n_list, wcss_scores, s=100, c='blue', marker='o', edgecolors='black')
    plt.ylabel('WCSS')
    plt.xticks(n_list)
    plt.xlabel('N_clusters')
    ax_2=plt.gca().twinx()
    create_elbow_plot(n_list,wcss_scores,ax_2)
    plt.legend()

    plt.tight_layout()

    #Plotting BIC
    fig=plt.figure()
    fig.set_size_inches(10,5)

    plt.title('BIC Score & Clusters')
    plt.plot(n_list,BIC_scores)
    plt.scatter(n_list, BIC_scores, s=100, c='blue', marker='o', edgecolors='black')
    plt.xticks(n_list)
    plt.xlabel('N_clusters')
    plt.ylabel('BIC Score')
    plt.tight_layout()

    #Plotting AIC
    fig=plt.figure()
    fig.set_size_inches(10,5)

    plt.title('AIC Score & Clusters')
    plt.plot(n_list,AIC_scores)
    plt.scatter(n_list, AIC_scores, s=100, c='blue', marker='o', edgecolors='black')
    plt.ylabel('AIC Score')
    plt.xticks(n_list)
    plt.xlabel('N_clusters')
    ax_2=plt.gca().twinx()
    create_elbow_plot(n_list,AIC_scores,ax_2)
    plt.legend()

    plt.tight_layout()

def experiment_km_clusters(dataset,dataset_name,labels_og,algorithm_name,cluster_graphs=4,categorical_cols=None):
    n_list=np.arange(1,20)
    data_categorized=categorizer(dataset,categorical_cols)

    #Converting dataset to 2D so it's plottable
    converter=TSNE(n_components=2)
    dataset_2d=converter.fit_transform(data_categorized)
    silhouette_scores=[]
    all_labels=[]
    wcss_scores=[]
    KM_dict={}
    KM_cost=[]

    #Assigning Clusters
    for n in tqdm(n_list):
        #Creating Gaussian
        km = KModes(n_clusters=n, init='Huang', n_init=3, verbose=0)

        #Fitted labels dataset
        km.fit(data_categorized,categorical=np.arange(data_categorized.shape[-1]))

        #Predictions
        labels=km.predict(data_categorized,categorical=np.arange(data_categorized.shape[-1]))
        all_labels.append(labels)
        
        #Calculating Mutal Info Score
        if n==1:
            pass
        else:
            silhouette_scores.append(silhouette_score(data_categorized,labels))

        #WCSS Score
        wcss_scores.append(calculate_wcss(data_categorized,labels,km.cluster_centroids_))
        
        #Cost
        KM_cost.append(km.cost_)
        
        KM_dict[n]=km

    #Plotting 2d color coded with labels 
    fig,axes=plt.subplots(1,len(n_list[:cluster_graphs])+1)
    fig.set_size_inches(35,5)

    plt.suptitle(f'{dataset_name} dataset clusters using {algorithm_name}',fontsize=18)

    for c,ax in enumerate(fig.axes):

        if c>-1:
            #Predictions
            labels=all_labels[c+1]

            #Plotting
            sc=ax.scatter(dataset_2d[:,0],dataset_2d[:,1],c=labels,cmap='plasma')
            #ax.legend()
            ax.set_ylabel('Y');
            ax.set_xlabel('X');
            ax.set_title('{} : COMPONENTS'.format(n_list[c+1]));
            ax.legend(*sc.legend_elements(), title='clusters')

        else:

            sc=ax.scatter(dataset_2d[:,0],dataset_2d[:,1],c=labels_og,cmap='plasma')
            #ax.legend()
            ax.set_ylabel('Y');
            ax.set_xlabel('X');
            ax.set_title('Dataset');      
            ax.legend(*sc.legend_elements(), title='clusters')

    plt.tight_layout()

    #Plotting WCSS
    fig=plt.figure()
    fig.set_size_inches(10,5)

    plt.title('WCSS & Clusters')
    plt.plot(n_list,wcss_scores)
    plt.scatter(n_list, wcss_scores, s=100, c='blue', marker='o', edgecolors='black')
    plt.xticks(n_list)
    plt.xlabel('N_clusters')
    plt.ylabel('WCSS')
    ax_2=plt.gca().twinx()
    create_elbow_plot(n_list,wcss_scores,ax_2)
    plt.legend()

    plt.tight_layout()

    #Plotting Silhouette score
    fig=plt.figure()
    fig.set_size_inches(10,5)

    plt.title('Silhouette_scores & Clusters')
    plt.plot(n_list[1:],silhouette_scores)
    plt.scatter(n_list[1:], silhouette_scores, s=100, c='blue', marker='o', edgecolors='black')
    plt.xlabel('N_clusters')
    plt.ylabel('Silhouette_scores')

    plt.tight_layout()

    #Plot Cost
    fig=plt.figure()
    fig.set_size_inches(10,5)

    plt.title('Cost & Clusters')
    plt.plot(n_list,KM_cost)
    plt.scatter(n_list, KM_cost, s=100, c='blue', marker='o', edgecolors='black')
    plt.xticks(n_list)
    plt.xlabel('N_clusters')
    plt.ylabel('Cost')
    ax_2=plt.gca().twinx()
    create_elbow_plot(n_list,KM_cost,ax_2)
    plt.legend()


    plt.tight_layout()