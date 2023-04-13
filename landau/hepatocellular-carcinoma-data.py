# hepatocellular-carcinoma-data.py
#
# Bryan Daniels
# 2023/3/31
#
# Apply our technique to liver cancer dataset (GEO GSE6764)
#
# Forked from hepatocellular-carcinoma-data.ipynb


import pandas as pd
import numpy as np

from landau.landauAnalysis import principalComponents
from landau import landauAnalysis

from toolbox.simplePickle import save

def load_data():
    expr = pd.read_csv('Data/230329/GSE6764.csv',index_col=0).T
    groups = pd.read_csv('Data/230329/GSE6764_groups.csv')['x']
    gene_symbols = pd.read_csv('Data/230329/GSE6764_gene_symbols.csv')['x']
    gene_functions = pd.read_csv('Data/230329/GSE6764_gene_GO_functions.csv')['x']

    # create gene info dataframe indexed by ID
    genedf = pd.DataFrame(np.array([gene_symbols,gene_functions]).T,
                          columns=['gene symbol','GO function'],
                          index=expr.columns)

    # create expression dataframe indexed by group
    groups.index = expr.index
    expr['group'] = groups
    expr_by_group = expr.set_index('group')
    
    # check that group memberships line up correctly
    assert(expr.group['GSM155919']=='control')
    assert(expr.group['GSM155945']=='early.HCC')
    assert(expr.group['GSM155993']=='very.early.HCC')
    
    return expr_by_group,genedf

def analyze_multiple_groups(expr_by_group,groups_to_include_list,
    skip_factor=1):

    # loop over group sets
    resultsDict = {}
    for groups_to_include in groups_to_include_list:
        print("Analyzing group(s) {}...".format(groups_to_include))
        
        data_to_use = expr_by_group.loc[groups_to_include,::skip_factor]
        vals,vecs = principalComponents(data_to_use,k=1)
        reduced_expr = pd.DataFrame(np.dot(data_to_use-data_to_use.mean(),vecs[0]),
                                index=data_to_use.index)
        
        landauData = landauAnalysis.landauAnalysis(data_to_use)
        
        # compute s values
        pc = pd.Series(vecs[0], index=data_to_use.columns)
        variances = pd.Series(np.diag(data_to_use.cov(ddof=0)), index=data_to_use.columns)
        Jnu = landauData['valList'][0]
        s = pc**2 / Jnu / variances
        s.name = 's'
        
        resultsDict[groups_to_include] = {'vals': vals,
                                          'vecs': vecs,
                                          'reduced_expr': reduced_expr,
                                          'landauData': landauData,
                                          'bicDiff': landauData['bicDiffList'][0],
                                          's': s,}
    return resultsDict
        
    
if __name__ == '__main__':

    # load data
    expr_by_group,genedf = load_data()

    # define sets of groups to test
    sorted_groups = ['control',
                     'cirrhosis',
                     'low.grade.dysplastic',
                     'high.grade.dysplastic',
                     'very.early.HCC',
                     'early.HCC',
                     'advanced.HCC',
                     'very.advanced.HCC']
    individual_groups = [ (group,) for group in sorted_groups ]
    paired_with_control_groups = [ ('control',group) for group in sorted_groups[1:] ]
    groups_to_include_list = individual_groups + paired_with_control_groups

    skip_factor = 1

    # run analysis
    resultsDict = analyze_multiple_groups(expr_by_group,
                                          groups_to_include_list,
                                          skip_factor=skip_factor)
                                          
    # write to file
    save(resultsDict,
         '230331_HCC_resultsDict_skip{}.pkl'.format(skip_factor))
