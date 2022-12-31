import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os 
import cv2
from glob  import glob



def prepare_data(df):
    df['case'] = df['id'].apply(lambda x: x.split("_")[0].replace("case",""))
    df['day'] = df["id"].apply(lambda x: x.split("_")[1].replace("day",""))
    # df['slice'] = df["id"].apply(lambda x: x.split("_")[3])

    list_images = glob('train/*/*/*/*.png')

    tmp_df = pd.DataFrame([ (item, item.split("/")[2], item.split("/")[4]) for item  in list_images ],columns=['full_path', 'case_and_day', 'img_file_name'])

    
    
    tmp_df['slice'] = tmp_df['img_file_name'].apply(lambda x: f"slice_{x.split('_')[1]}")
    tmp_df['height'] = tmp_df['img_file_name'].apply(lambda x: int(x.split("_")[2]))
    tmp_df['width'] = tmp_df['img_file_name'].apply(lambda x: int(x.split("_")[3]))
    tmp_df['id'] = tmp_df['case_and_day']+"_"+tmp_df['slice']

    df = pd.merge(df, tmp_df,  on='id', how='left')

    del list_images, tmp_df

    return df

def rearrange_dataframe_for_3_segmentation_classes(df):
    df_restructured = pd.DataFrame({'id': df['id'][::3]})
    
    df_restructured['large_bowel'] = df['segmentation'][::3].values
    df_restructured['small_bowel'] = df['segmentation'][1::3].values
    df_restructured['stomach'] = df['segmentation'][2::3].values

    df_restructured['full_path'] = df['full_path'][::3].values
    df_restructured['case'] = df['case'][::3].values
    df_restructured['day'] = df['day'][::3].values
    df_restructured['slice'] = df['slice'][::3].values
    df_restructured['width'] = df['width'][::3].values
    df_restructured['height'] = df['height'][::3].values

    df_restructured = df_restructured.reset_index(drop=True)
    df_restructured = df_restructured.fillna("")

    df_restructured['count'] = np.sum(df_restructured.iloc[:, 1:4] !=  "", axis= 1).values

    return df_restructured









