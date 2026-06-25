import os
import time
import numpy as np
import pandas as pd

import config
import pickle
from experimentor import DataCont
from sklearn.model_selection import train_test_split
import torch
import openpyxl

def load_MIDA(data_type) -> DataCont:

    start_time = time.time()

    if data_type == 'ibd':
        count_name = 'ibd_count.csv'
        phylo_name = 'ibd_phylo.csv'
    else:
        count_name = 'vag_count.csv'
        phylo_name = 'vag_phylo.csv'
        

    raw_data = pd.read_csv(os.path.join('data/MIDA/', count_name), index_col = 0)
    phylo = pd.read_csv(os.path.join('data/MIDA/', phylo_name), index_col = 0)

    labels = np.ones(raw_data.shape[0])
    dataset = raw_data.values.astype(np.float64)

    dc = DataCont(X_train= dataset, y_train =labels, phylo = phylo)

    print(f"--- Loaded in {round(time.time() - start_time, 2)} seconds ---")
    
    return (dc)


def load_MBGAN() -> DataCont:
    
    start_time = time.time()
    train_data_filename= config.c_raw_data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_filepath = os.path.join(script_dir, 'data/MBGAN', train_data_filename)

    raw_data = pickle.load(open(train_data_filepath, 'rb'))
    dataset = raw_data.iloc[:,1:].values/100
    labels = raw_data["group"]
    taxa_list = raw_data.columns[1:]
    phylo = pd.DataFrame(taxa_list).T

    label_dict = {
        
            'ctrl': 0,
            'case': 1
        }

    labels = labels.replace(label_dict).values.astype(np.float64)
    dataset = dataset.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    dc = DataCont(X_train= X_train, X_test=X_test, y_train=y_train, y_test=y_test, phylo = phylo)

    print(f"--- Loaded in {round(time.time() - start_time, 2)} seconds ---")
    
    return (dc)

def load_deepmicro(data_key):

    # Dynamically get the filename from the config module based on the data_key
    try:
        filename = getattr(config, data_key)
    except AttributeError:
        raise ValueError(f"Dataset key '{data_key}' not found in config.")
        
    study = "deepmicro"
    file_path = os.path.join('data', study, filename)
    raw_data = pd.read_csv(file_path, sep='\t', index_col=0, dtype='str').T
    
    unique_conditions = sorted([condition for condition in raw_data['disease'].unique()]) 
    #Create label dictionary with 'control' always mapped to 0
    label_dict = {}
    # Assign incremental labels starting from 1 to other conditions
    label_dict.update({condition: i+1 for i, condition in enumerate(unique_conditions)})
    label_dict_df = pd.DataFrame(list(label_dict.items()), columns=['Type', 'label'])
    
    first_k_index = next((index for index, col in enumerate(raw_data.columns) if col.startswith('k_')), len(raw_data.columns))
    # List all column names before the first 'k_' column
    columns_before_first_k = raw_data.columns[:first_k_index]
    exclude = columns_before_first_k
    # Get all column names excluding 'study_condition'
    phylo = pd.DataFrame(raw_data.columns[~raw_data.columns.isin(exclude)]).T.reset_index(drop = True)
    
    label = raw_data['disease']
    y_train = label.replace(label_dict).values.astype(np.float64)
    X_train = raw_data.drop(columns = exclude).values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2,  stratify=label, random_state=42)

    dc = DataCont(X_train= X_train, X_test = X_test, y_train= y_train, y_test = y_test, label_dict = label_dict_df, phylo = phylo)

    return(dc)

def load_gangmhi(train_batch, test_batch) -> DataCont:
    
    start_time = time.time()

    study = 'gmhi'

    if train_batch == 'all':
        trbatch = ['Jie (2017)', 'Feng (2015)', 'Zeller (2014)', 'Vogtmann (2016)', 'Yu (2015)', 'Nielsen (2014)', 
                   'Hall (2017)', 'Schirmer (2018)', 'He (2017)', 'Rampeli (2015)', 'Qin (2012)', 'Karlsson (2013)','Qin (2014)', 'Liu (2017)', 'Tanca (2017)', 'Petersen (2017)',
                   'Obregon-Tito (2015)', 'Lim (2014)', 'Schirmer (2016)', 'Zeevi (2015)', 'Zhang (2015)', 'Sankaranarayanan (2015)', 'Nishijima (2016)', 'Raymond (2015)', 'Li (2017)',
                   'Backhed (2015)', 'Le Chatelier (2013)', 'Karlsson (2012)',
                   'Guthrie (2017)', 'Liu (2016)', 'HMP1: Huttenhower (2012) and HMP2: Lloyd-Price (2017)', 'Louis (2016)', 'Palleja (2016)']
    else:
        trbatch = train_batch

    if test_batch == 'all':
        valbatch = ['Qin (2014)', 'Bedarf (2017)', 'Vaughn (2016)', 'Wen (2017)', 'Loomba (2017)', 'Thomas (2019)', 'Wirbel (2019)', 'Dhakan (2019)','This study']
    else:
        valbatch = test_batch
    
    ab_filename = config.c_4347_final_relative_abundances
    meta_filename = config.c_Final_metadata_4347

    ab_filepath = os.path.join(os.getcwd(), 'data', study, ab_filename)
    train = pd.read_csv(ab_filepath, sep= '\t', index_col = 0, low_memory=False).T

    meta_filepath = os.path.join(os.getcwd(), 'data', study, meta_filename)
    metadata = pd.read_csv(meta_filepath, index_col = 0, low_memory=False).T

    ab_filename = config.c_validation_abundance
    meta_filename = config.c_validation_metadata1

    ab_filepath = os.path.join(os.getcwd(), 'data', study, ab_filename)
    val = pd.read_csv(ab_filepath, index_col = 0).T

    meta_filepath = os.path.join(os.getcwd(), 'data', study, meta_filename)
    metadata_val = pd.read_csv(meta_filepath, index_col = 0)

    names_df1 = train.columns  
    names_df2 = val.columns

    common_names = names_df1.intersection(names_df2)

    train = train[common_names.intersection(train.columns)]
    val = val[common_names.intersection(val.columns)]

    phylo = pd.DataFrame(train.columns).T

    train.insert(0,'Batch',metadata['Author (year)'].tolist())
    train.insert(0,'Type','')
    train['Type']=pd.Series(train.index.values).str.split('_',expand=True)[0].tolist()

    raw_data_train = train[train['Batch'].isin(trbatch)]
    
    pd.set_option('display.max_colwidth', None)
    summary_train = raw_data_train.groupby('Batch')['Type'].unique().reset_index()
    summary_train['Sample Count'] = raw_data_train.groupby('Batch')['Type'].count().values
    
    # Get all column names excluding 'study_condition'
    
    exclude = ['Type', 'Batch']
    
    label = raw_data_train['Type']
    
    #Extract unique values sorted alphabetically for consistent label assignment (excluding 'control' to manually assign it later)
    unique_conditions = sorted([condition for condition in raw_data_train['Type'].unique()]) 
    #Create label dictionary with 'control' always mapped to 0
    label_dict = {}
    # Assign incremental labels starting from 1 to other conditions
    label_dict.update({condition: i+1 for i, condition in enumerate(unique_conditions)})
    label_dict_df = pd.DataFrame(list(label_dict.items()), columns=['Type', 'label'])
    
    y_train = label.replace(label_dict).values.astype(np.float64)
    X_train = raw_data_train.drop(columns = exclude).values.astype(np.float64)

    ##### for validation #####

    val.insert(0,'Batch',metadata_val['Batch'].tolist())
    val.insert(0,'Type','')
    val['Type']=metadata_val['Phenotype_all2'].tolist()

    raw_data_val = val[val['Batch'].isin(valbatch)]
    
    pd.set_option('display.max_colwidth', None)
    summary_val = raw_data_val.groupby('Batch')['Type'].unique().reset_index()
    summary_val['Sample Count'] = raw_data_val.groupby('Batch')['Type'].count().values
    
    # Get all column names excluding 'study_condition'
    
    exclude = ['Type', 'Batch']
    phylo_val = pd.DataFrame(raw_data_val.drop(columns= exclude).columns).T
    
    label = raw_data_val['Type']
    
    #Extract unique values sorted alphabetically for consistent label assignment (excluding 'control' to manually assign it later)
    unique_conditions = sorted([condition for condition in raw_data_val['Type'].unique()]) 
    #Create label dictionary with 'control' always mapped to 0
    label_dict = {}
    # Assign incremental labels starting from 1 to other conditions
    label_dict.update({condition: i+1 for i, condition in enumerate(unique_conditions)})
    label_dict_df = pd.DataFrame(list(label_dict.items()), columns=['Type', 'label'])
    
    y_test = label.replace(label_dict).values.astype(np.float64)
    X_test = raw_data_val.drop(columns = exclude).values.astype(np.float64)
    
    dc = DataCont(X_train= X_train, X_test = X_test, y_train= y_train, y_test = y_test, label_dict = label_dict_df, phylo = phylo)
    
    print(f"--- Loaded in {round(time.time() - start_time, 2)} seconds ---")
    
    return(dc)

def load_giliberti(data_key, data_type) -> DataCont:
    
    start_time = time.time()
    
    # Dynamically get the filename from the config module based on the data_key
    try:
        data_filename = getattr(config, data_key)
    except AttributeError:
        raise ValueError(f"Dataset key '{data_key}' not found in config.")

    if data_type == '16s':

        data_filepath = os.path.join(os.getcwd(), 'data', 'giliberti', '16s', data_filename)
        raw_data = pd.read_csv(data_filepath, sep='\t', index_col=0, dtype='str').T
        unique_conditions = sorted([condition for condition in raw_data['study_condition'].unique() if condition != 'H'])
        #Create label dictionary with 'control' always mapped to 0
        label_dict = {'H': 0}

    else:
        data_filepath = os.path.join(os.getcwd(), 'data', 'giliberti', 'meta', data_filename)
        raw_data = pd.read_csv(data_filepath, sep='\t', index_col=0, dtype='str').T
        unique_conditions = sorted([condition for condition in raw_data['study_condition'].unique() if condition != 'control'])
        #Create label dictionary with 'control' always mapped to 0
        label_dict = {'control': 0}

    # Assign incremental labels starting from 1 to other conditions
    label_dict.update({condition: i+1 for i, condition in enumerate(unique_conditions)})
    label_dict_df = pd.DataFrame(list(label_dict.items()), columns=['study_condition', 'label'])
    first_k_index = next((index for index, col in enumerate(raw_data.columns) if col.startswith('k_')), len(raw_data.columns))
    # List all column names before the first 'k_' column
    columns_before_first_k = raw_data.columns[:first_k_index]
    exclude = columns_before_first_k
    # Get all column names excluding 'study_condition'
    phylo = pd.DataFrame(raw_data.columns[~raw_data.columns.isin(exclude)]).T.reset_index(drop = True)

    label = raw_data['study_condition']
    y_train = label.replace(label_dict).values.astype(np.float64)
    X_train = raw_data.drop(columns = exclude).values.astype(np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2,  stratify=y_train, random_state=42)

    dc = DataCont(X_train= X_train, X_test = X_test, y_train= y_train, y_test = y_test, label_dict = label_dict_df, phylo = phylo)
    
    print(f"--- Loaded in {round(time.time() - start_time, 2)} seconds ---")

    return(dc)

def load_FeMAI(validation = 'PRJNA763023') -> DataCont:

    start_time = time.time()

    # get metadata and conditions

    file_dir = 'data/FeMAI'
    file_name = "metadata_2340_CRC_cohort_20240426.xlsx"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_file = os.path.join(script_dir, file_dir, file_name)
    metadata = pd.read_excel(metadata_file, sheet_name = 2, engine = 'openpyxl')
    metadata.index = metadata.iloc[:,0]
    metadata.index.name = None
    metadata = metadata.drop(metadata.columns[0], axis = 1)
    exclude = metadata[metadata['excluded'] == 'yes'].index
    adenoma = metadata[(metadata['class'] == 'adenoma') | (metadata['class'].isna())].index
    comb_ind = exclude.union(adenoma)

    study_dict = pd.read_excel(metadata_file, sheet_name = 0, header = 4, engine = 'openpyxl')
    study_dict = study_dict[0:17]
    study_dict = study_dict.set_index('BioProject')['paper'].to_dict()

    file_name = "species_signal_count_table_2340_CRC_cohort_20240322.tab"
    count_file = os.path.join(script_dir, file_dir, file_name)

    # exclude from metadata and columns and rows with all 0
    #format
    species_count = pd.read_csv(count_file, sep='\t', header = None)
    species_count = species_count.T
    species_count.columns = species_count.iloc[0]
    species_count = species_count.drop(species_count.index[0])
    species_count.index = species_count.iloc[:,0]
    species_count.index.name = None
    species_count = species_count.drop(species_count.columns[0], axis = 1)
    species_count = species_count.drop(index = comb_ind)
    species_count = species_count.astype(float)

    # filter by presence threshold
    num_samples = species_count.shape[0]
    presence_counts = (species_count > 0).sum(axis=0)
    threshold = 0.05 * num_samples
    species_count = species_count.loc[:, presence_counts >= threshold]
    zeros = (species_count == 0).all(axis = 1)
    species_count = species_count[~zeros]

    # get relative abundance table
    relative_abundance = species_count.div(species_count.sum(axis=1), axis=0)

    meta_rel = metadata[metadata.index.isin(relative_abundance.index)]
    test_index = meta_rel[meta_rel['study_accession'] == validation].index #PRJEB12449
    train_index = meta_rel[meta_rel['study_accession'] != validation].index

    disease_label = meta_rel['class']

    label_dict = {
        'healthy': 0,
        'CRC': 1
    }

    disease_label = disease_label.replace(label_dict)

    label_dict = {value: key for key, value in label_dict.items()}

    train_rel = relative_abundance.loc[train_index]
    test_rel = relative_abundance.loc[test_index]

    train_lab = disease_label.loc[train_index]
    test_lab = disease_label.loc[test_index]

    # get phylo
    file_name = "species_taxo_phylo_20240322.tab"
    phylo_file = os.path.join(script_dir, file_dir, file_name)

    phylo = pd.read_csv(phylo_file, sep='\t', header = None)

    df = phylo.copy()

    for col in phylo.columns[2:]:
        first_letter = phylo.iloc[0, col][0].lower()
        df.iloc[1:, col] = first_letter + '_' + df.iloc[1:, col].astype(str).str.replace(' ', '_')

    df = df.T
    df.columns = df.iloc[0,:]
    df = df.iloc[1:,1:]
    concatenated = df.iloc[-2:0:-1].apply(lambda col: '|'.join(col.astype(str)), axis=0)
    df_phylo = pd.DataFrame([concatenated], columns = df.columns)
    common_columns = df_phylo.columns.intersection(relative_abundance.columns)
    phylo = df_phylo[common_columns]

    X_train = torch.tensor(train_rel.values)
    y_train = torch.tensor(train_lab.values)
    X_test = torch.tensor(test_rel.values)
    y_test = torch.tensor(test_lab.values)

    dc = DataCont(X_train= X_train, X_test = X_test, y_train= y_train, y_test = y_test, label_dict = label_dict, phylo = phylo)

    print(f"--- Loaded in {round(time.time() - start_time, 2)} seconds ---")
        
    return(dc)

def load_clooney_ext(cnt = False) -> DataCont:

    if cnt == True:
        train_data_filename= config.clooney_train_cnt; train_label_filename  = config.clooney_train_label_cnt; test_data_filename= config.clooney_test_cnt; test_label_filename  = config.clooney_test_label_cnt; phylo_data_filename= config.clooney_ext_phylo_cnt
    else:
        train_data_filename= config.clooney_train; train_label_filename  = config.clooney_train_label; test_data_filename= config.clooney_test; test_label_filename  = config.clooney_test_label; phylo_data_filename= config.clooney_ext_phylo
    start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_filepath = os.path.join(script_dir, 'data', train_data_filename)
    test_data_filepath = os.path.join(script_dir, 'data', test_data_filename)
    train_label_filepath = os.path.join(script_dir, 'data', train_label_filename)
    test_label_filepath = os.path.join(script_dir, 'data', test_label_filename)
    phylo_data_filepath = os.path.join(script_dir, 'data', phylo_data_filename)
    
    train = pd.read_csv(train_data_filepath, index_col=0)
    
    phylo = pd.read_csv(phylo_data_filepath, index_col = 0)
    phylo.reset_index(drop=True, inplace=True)
    
    common_columns = train.columns.intersection(phylo.columns)
    
    phylo = phylo[common_columns]
    train = train.values.astype(np.float64)
    test = pd.read_csv(test_data_filepath, index_col=0).values.astype(np.float64)
    
    label_dict = {
        # Controls
        'HC': 0,
        'CD': 1,  # 0
        'UC': 2 #
    }
    
    
    lab1 = pd.read_csv(train_label_filepath, index_col=0)
    
    lab1 = lab1.replace(label_dict)
    train_y = lab1.values.astype(np.float64)
    
    lab2 = pd.read_csv(test_label_filepath, index_col=0)
    
    lab2 = lab2.replace(label_dict)
    test_y = lab2.values.astype(np.float64)

    label_dict = {value: key for key, value in label_dict.items()}
    
    dc = DataCont(X_train=train, X_test=test, y_train=train_y, y_test=test_y, phylo = phylo, label_dict = label_dict)
    
    print(f"--- Loaded in {round(time.time() - start_time, 2)} seconds ---")
    return(dc)



