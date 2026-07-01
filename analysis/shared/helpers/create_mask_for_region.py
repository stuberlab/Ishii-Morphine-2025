import numpy as np
import pandas as pd


def create_mask_for_region(atlas_id,atlasimg,atlas_df,include_subregions = False):
    # function that creates a mask from the atlasimg
    # labelimg: image file which stores brain region label for each pixel in atlas space
    # atlas_df: a data frame containing meta information of brain atlas. this dataframe should contain "id" and "parent_id" column.
    # include_subregions: if include_subregions is true, all child regions of the atlas_id 

    # create a tmp mask array
    tmpmask = np.zeros(np.shape(atlasimg),dtype = 'bool')

    try:
        if include_subregions:
            subset_df = get_subregions(atlas_df, atlas_id)
            if len(subset_df) == 0:
                tmpmask[np.where(atlasimg == atlas_id)] = True
            else:
                tmpmask[np.where(np.isin(atlasimg, np.append(subset_df['id'].values,atlas_id)))] = True
        else:
            tmpmask[np.where(atlasimg == atlas_id)] = True
    except:
        print("No atlas_id in the atlas dataframe")
    return tmpmask

def get_subregions(dataframe, region_id, collected=None,return_original = False):
    """
    Recursively get all subregions of a given region_id from the dataframe.

    Args:
        dataframe (pandas.DataFrame): DataFrame containing brain region data.
        region_id (int): ID of the region to get subregions for.
        collected (list, optional): List to store collected subregions. Defaults to None.

    Returns:
        list: List of dictionaries representing subregions.
    """
    if collected is None:
        collected = []


    # Find subregions of the current region_id
    subregions = dataframe[dataframe['parent_id'] == region_id].to_dict('records')

    # Add current subregions to the collected list
    collected.extend(subregions)

    # Recursively search for subregions of each subregion
    for subregion in subregions:
        get_subregions(dataframe, subregion['id'], collected)

    if return_original:
        original = dataframe[dataframe['id'] == region_id].to_dict('records')
        collected.extend(original)
        return_original = False
        
    return pd.DataFrame(collected)
