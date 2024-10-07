from libs.prefect_helpers import *
from libs.data_manipulation import *
from libs.feature_generation import *
from libs.dim_reduction import *
from libs.football_plots import *
from libs.clustering import *
from prefect import task, flow

@flow
def ripleys_k_flow(name : str):
    df : pd.DataFrame = task_wrapper(compile_team_tracking_data, use_cache=True)("data", name)
    df = task_wrapper(extract_one_match,use_cache=False)(df,1)
    print(len(df))
    ripleys_k_vals = task_wrapper(ripley_k_by_indices, use_cache=True)(df, df.iloc[::48].index)
    pca_obj = PCAObject(ripleys_k_vals, n_components=10)
    np_pca = task_wrapper(pca_obj.transform,use_cache=True)(ripleys_k_vals)
    plt.scatter(np_pca[:,0], np_pca[:,1])
    plt.savefig(name + "_ripleysk")
    clusterer = KMeansObject(np_pca, 5)
    labels = clusterer.get_labels()

    plt.scatter(np_pca[:,0], np_pca[:,1], labels = labels)
    plt.savefig(name + "_ripleysk_clustered")



ripleys_k_flow("Denmark")