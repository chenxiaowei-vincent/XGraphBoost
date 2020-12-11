import pandas as pd
import numpy as np

max_depth_numbers = [2, 4, 6, 8, 10]
learning_rate_numbers = [0.01, 0.05, 0.1, 0.15, 0.2]
min_child_weight_numbers = [2, 4, 6, 8, 10]
names = locals()

def data_deal_regre(proteins):
    for protein in proteins:
        df = pd.read_csv(protein+'_scores.csv')
        df_all = pd.DataFrame()
        for min_child_weight_number in min_child_weight_numbers:
            names['df' + str(min_child_weight_number)] = pd.DataFrame(np.random.rand(25).reshape(5,5))
            for i,max_depth_number in enumerate(max_depth_numbers):
                for j,learning_rate_number in enumerate(learning_rate_numbers):
                    names['df' + str(min_child_weight_number)][i][j] = df.loc[(df.type=='dmpnn+xgb')&(df.max_depth==str(max_depth_number))&(df.learning_rate==str(learning_rate_number))&(df.min_child_weight==str(min_child_weight_number)),'MAE'].values[0]
            df_all = pd.concat([df_all,names['df' + str(min_child_weight_number)]],axis=1)
        df_all.to_csv('data_integration/'+protein+'_MAE.csv',index=None,header=None)
    for protein in proteins:
        df = pd.read_csv(protein+'_scores.csv')
        df_all = pd.DataFrame()
        for min_child_weight_number in min_child_weight_numbers:
            names['df' + str(min_child_weight_number)] = pd.DataFrame(np.random.rand(25).reshape(5,5))
            for i,max_depth_number in enumerate(max_depth_numbers):
                for j,learning_rate_number in enumerate(learning_rate_numbers):
                    names['df' + str(min_child_weight_number)][i][j] = df.loc[(df.type=='dmpnn+xgb')&(df.max_depth==str(max_depth_number))&(df.learning_rate==str(learning_rate_number))&(df.min_child_weight==str(min_child_weight_number)),'RMSE'].values[0]
            df_all = pd.concat([df_all,names['df' + str(min_child_weight_number)]],axis=1)
        df_all.to_csv('data_integration/'+protein+'_RMSE.csv',index=None,header=None)


def data_deal_classi(proteins):
    for protein in proteins:
        df = pd.read_csv(protein + '_scores.csv')
        df_all = pd.DataFrame()
        for min_child_weight_number in min_child_weight_numbers:
            names['df' + str(min_child_weight_number)] = pd.DataFrame(np.random.rand(25).reshape(5, 5))
            for i, max_depth_number in enumerate(max_depth_numbers):
                for j, learning_rate_number in enumerate(learning_rate_numbers):
                    if protein == 'clintox' or protein == 'tox21' or protein == 'sider' or protein == 'toxcast':
                        names['df' + str(min_child_weight_number)][i][j] = df.loc[
                            (df.type == 'dmpnn+xgb') & (df.max_depth == max_depth_number) & (
                                        df.learning_rate == learning_rate_number) & (
                                        df.min_child_weight == min_child_weight_number), 'auc'].values[0]
                    else:
                        names['df' + str(min_child_weight_number)][i][j] = df.loc[
                            (df.type == 'dmpnn+xgb') & (df.max_depth == str(max_depth_number)) & (
                                        df.learning_rate == str(learning_rate_number)) & (
                                        df.min_child_weight == str(min_child_weight_number)), 'auc'].values[0]
            df_all = pd.concat([df_all, names['df' + str(min_child_weight_number)]], axis=1)
        df_all.to_csv('data_integration/'+protein + '_auc.csv', index=None, header=None)
    for protein in proteins:
        df = pd.read_csv(protein + '_scores.csv')
        df_all = pd.DataFrame()
        for min_child_weight_number in min_child_weight_numbers:
            names['df' + str(min_child_weight_number)] = pd.DataFrame(np.random.rand(25).reshape(5, 5))
            for i, max_depth_number in enumerate(max_depth_numbers):
                for j, learning_rate_number in enumerate(learning_rate_numbers):
                    if protein == 'clintox' or protein == 'tox21' or protein == 'sider' or protein == 'toxcast':
                        names['df' + str(min_child_weight_number)][i][j] = df.loc[
                            (df.type == 'dmpnn+xgb') & (df.max_depth == max_depth_number) & (
                                        df.learning_rate == learning_rate_number) & (
                                        df.min_child_weight == min_child_weight_number), 'sp'].values[0]
                    else:
                        names['df' + str(min_child_weight_number)][i][j] = df.loc[
                            (df.type == 'dmpnn+xgb') & (df.max_depth == str(max_depth_number)) & (
                                        df.learning_rate == str(learning_rate_number)) & (
                                        df.min_child_weight == str(min_child_weight_number)), 'sp'].values[0]
            df_all = pd.concat([df_all, names['df' + str(min_child_weight_number)]], axis=1)
        df_all.to_csv('data_integration/'+protein + '_sp.csv', index=None, header=None)
    for protein in proteins:
        df = pd.read_csv(protein + '_scores.csv')
        df_all = pd.DataFrame()
        for min_child_weight_number in min_child_weight_numbers:
            names['df' + str(min_child_weight_number)] = pd.DataFrame(np.random.rand(25).reshape(5, 5))
            for i, max_depth_number in enumerate(max_depth_numbers):
                for j, learning_rate_number in enumerate(learning_rate_numbers):
                    if protein == 'clintox' or protein == 'tox21' or protein == 'sider' or protein == 'toxcast':
                        names['df' + str(min_child_weight_number)][i][j] = df.loc[
                            (df.type == 'dmpnn+xgb') & (df.max_depth == max_depth_number) & (
                                    df.learning_rate == learning_rate_number) & (
                                    df.min_child_weight == min_child_weight_number), 'sn'].values[0]
                    else:
                        names['df' + str(min_child_weight_number)][i][j] = df.loc[
                            (df.type == 'dmpnn+xgb') & (df.max_depth == str(max_depth_number)) & (
                                    df.learning_rate == str(learning_rate_number)) & (
                                    df.min_child_weight == str(min_child_weight_number)), 'sn'].values[0]
            df_all = pd.concat([df_all, names['df' + str(min_child_weight_number)]], axis=1)
        df_all.to_csv('data_integration/'+protein + '_sn.csv', index=None, header=None)
    for protein in proteins:
        df = pd.read_csv(protein + '_scores.csv')
        df_all = pd.DataFrame()
        for min_child_weight_number in min_child_weight_numbers:
            names['df' + str(min_child_weight_number)] = pd.DataFrame(np.random.rand(25).reshape(5, 5))
            for i, max_depth_number in enumerate(max_depth_numbers):
                for j, learning_rate_number in enumerate(learning_rate_numbers):
                    if protein == 'clintox' or protein == 'tox21' or protein == 'sider' or protein == 'toxcast':
                        names['df' + str(min_child_weight_number)][i][j] = df.loc[
                            (df.type == 'dmpnn+xgb') & (df.max_depth == max_depth_number) & (
                                        df.learning_rate == learning_rate_number) & (
                                        df.min_child_weight == min_child_weight_number), 'acc'].values[0]
                    else:
                        names['df' + str(min_child_weight_number)][i][j] = df.loc[
                            (df.type == 'dmpnn+xgb') & (df.max_depth == str(max_depth_number)) & (
                                        df.learning_rate == str(learning_rate_number)) & (
                                        df.min_child_weight == str(min_child_weight_number)), 'acc'].values[0]
            df_all = pd.concat([df_all, names['df' + str(min_child_weight_number)]], axis=1)
        df_all.to_csv('data_integration/'+protein + '_acc.csv', index=None, header=None)

def morgan_xgb_result(proteins):
    for protein in proteins:
        df = pd.read_csv(protein + '_scores.csv')
        if protein == 'clintox' or protein == 'tox21' or protein == 'sider' or protein == 'bace' or protein == 'bbbp' or protein == 'hiv' or protein == 'toxcast':
            df1 = df.loc[
                (df.type == 'morgan+xgb') & (df.max_depth == str(6)) & (
                            df.learning_rate == str(0.01)) & (
                            df.min_child_weight == str(4))]

            df2 = df.loc[df.type == 'dmpnn']
            df3 = df.loc[
                (df.type == 'dmpnn+xgb') & (df.max_depth == str(6)) & (
                            df.learning_rate == str(0.01)) & (
                            df.min_child_weight == str(4))]
        else:
            df1 = df.loc[
                (df.type == 'morgan+xgb') & (df.max_depth == str(4)) & (
                            df.learning_rate == str(0.1)) & (
                            df.min_child_weight == str(8))]
            df2 = df.loc[df.type == 'dmpnn']
            df3 = df.loc[
                (df.type == 'dmpnn+xgb') & (df.max_depth == str(4)) & (
                            df.learning_rate == str(0.1)) & (
                            df.min_child_weight == str(8))]


        df_save = pd.concat([df1,df2])
        df_save = pd.concat([df_save,df3])
        try:
            df_save.columns = ['type','max_depth','learning_rate','min_child_weight','auc','sn','sp','acc']
            del df_save['max_depth'],df_save['learning_rate'],df_save['min_child_weight']
        except:
            df_save.columns = ['type','max_depth','learning_rate','min_child_weight','MAE', 'RMSE']
            del df_save['max_depth'],df_save['learning_rate'],df_save['min_child_weight']
        df_save.to_csv('data_integration/'+protein + '_result.csv', index=None)

if __name__ == '__main__':
    # data_deal_regre(['sampl_freesolv','delaney_ESOL','lipo'])
    # data_deal_classi(['bace','bbbp','hiv','clintox','tox21','sider'])
    # data_deal_classi(['toxcast'])
    morgan_xgb_result(['clintox'])
    # morgan_xgb_result(['toxcast'])
