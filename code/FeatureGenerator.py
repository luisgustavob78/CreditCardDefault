"""
Nesse script, vamos criar algumas classes para encapsular funções de criação de features.
Aqui, podemos notar que algumas funções realizam trasnformações de dados simples, mas o objetivo
de criá-las em formato de classe é para que possamos utilizá-las como step dentro de um sklearn 
pipeline, de forma a garantir que todo o nosso processamento de dados ficará dentro dos artefatos
a serem salvos do modelo.

Também podemos notar que algumas das classes possuem o método "fit" vazio, retornando apenas o "self".
Isso acontece pelo fato de que essas classes em específico não utilizam cálculos históricos como média,
máximo, mínimo, desvio padrão e etc e, portanto, não precisam se ater ao conjunto de treinamento para
serem utilizadas. Outras, no entanto, fazem uso desses cálculos e precisam ficar atreladas ao conjunto 
de treinamento para evitar leak de dados, por isso o método "fit" possui alguns cálculos nesse caso.
"""




import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

"""
===========================================
Classe para calcular divisão entre features
===========================================

Retorna um pandas dataframe com as colunas originais e as novas features calculadas.

:: Parâmetros 
    ** features_num (requerido): list
        Lista com o nome das features que serão usadas como numeradores da divisão.

    ** features_denom (requerido): list
        Lista com o nome das features que serão usadas como denominadores da divisão.

:: Exemplo:
-----------
    #Inicialização
    fbf = FeatureByFeature(features_num=["montante_pagamento_mes1", "montante_pagamento_mes2"],
                            features_denom=["montante_conta_mes1", "montante_conta_mes2"])

    #fit
    fbf.fit(df_train)

    #transform
    df_train = fbf.transform(df_train)
    df_val = fbf.transform(df_val)
    df_test = fbf.transform(df_test)
"""

class FeatureByFeature(BaseEstimator, TransformerMixin):

    def __init__(self, features_num, features_denom):
        self.features_num = features_num
        self.features_denom = features_denom

    def feature_calculator(self, data, f1, f2):
        for i in range(0, len(f1)):
            v1 = f1[i]
            v2 = f2[i]

            data[f"{v1}/{v2}"] = data[v1]/data[v2]

        return data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df_final = X.copy()
        df_final = self.feature_calculator(df_final, self.features_num, self.features_denom)
        return df_final

"""
===========================================
Classe para criar diferença entre features
===========================================

Retorna um pandas dataframe com as colunas originais e as novas features calculadas.

:: Parâmetros 
    ** features (requerido): list
        Lista com o nome das features que serão usadas como base para os cálculos de diferença.

:: Exemplo:
-----------
    #Inicialização
    diff_features = DiffFeatures(features=["montante_pagamento_mes1", "montante_pagamento_mes2"])

    #fit
    diff_features.fit(df_train)

    #transform
    df_train = diff_features.transform(df_train)
    df_val = diff_features.transform(df_val)
    df_test = diff_features.transform(df_test)
"""

class DiffFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        for i in range(1, len(self.features)):
            a = self.features[i-1]
            b = self.features[i]
            data[f"{b}_minus_{a}"] = b - a 

        return data

"""
=================================================
Classe para criar estatísticas entre agrupamentos
=================================================

Retorna um pandas dataframe com as colunas originais e as novas features calculadas.

:: Parâmetros
    ** group_columns (requerido): list
        Lista com o(s) nome da(s) coluna(s) que identifica(m) cada uma das múltiplas séries temporais.  
    ** features (requerido): list
        Lista com o nome das features que serão usadas como base para os cálculos estatísticos.

:: Exemplo:
-----------
    #Inicialização
    group_features = GroupFeatures(group_columns=["Forno"],
                        features=["temperatura", "pressao"])

    #fit
    group_features.fit(df_train)

    #transform
    df_train = group_features.transform(df_train)
    df_val = group_features.transform(df_val)
    df_test = group_features.transform(df_test)

:: OBSERVAÇÃO:
--------------
    Precisa que a divisão treino, validação e teste seja feita antes do uso dessa classe.
"""

class GroupFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, group_columns, features):
        self.group_columns = group_columns
        self.features = features
        
    def group_features(self, gp, vf, data):
        
        mean_names = []
        max_names = []
        min_names = []
        
        for f in vf:
            mean_name = f"{gp[0]}_{f}_mean"
            max_name = f"{gp[0]}_{f}_max"
            min_name = f"{gp[0]}_{f}_min"
            
            mean_names.append(mean_name)
            max_names.append(max_name)
            min_names.append(min_name)
            
        d_mean = dict(zip(vf, mean_names))
        d_max = dict(zip(vf, max_names))
        d_min = dict(zip(vf, min_names))
        
        df_group_mean = pd.DataFrame(data.groupby(gp)[vf[:]].mean().rename(columns=d_mean))
        df_group_mean = df_group_mean.reset_index()
        
        df_group_max = pd.DataFrame(data.groupby(gp)[vf[:]].max().rename(columns=d_max))
        df_group_max = df_group_max.reset_index()
        
        df_group_min = pd.DataFrame(data.groupby(gp)[vf[:]].min().rename(columns=d_min))
        df_group_min = df_group_min.reset_index()
        
        return df_group_mean, df_group_max, df_group_min
    
    def fit(self, X, y=None):
        self.X_group_mean, self.X_group_max, self.X_group_min = self.group_features(self.group_columns, self.features, X)
        return self.X_group_mean, self.X_group_max, self.X_group_min
                                    
    def transform(self, X):
                         
        #criando o dataframe
        X_final = X.merge(self.X_group_mean, on=self.group_columns, how="left")
        X_final = X_final.merge(self.X_group_max, on=self.group_columns, how="left")
        X_final = X_final.merge(self.X_group_min, on=self.group_columns, how="left")
                                    
        for value_feature in self.features:
            X_final[f"{value_feature}/{self.group_columns[0]}_mean"] = X_final[f"{value_feature}"] / X_final[f"{self.group_columns[0]}_{value_feature}_mean"]
            X_final[f"{value_feature}/{self.group_columns[0]}_max"] = X_final[f"{value_feature}"] / X_final[f"{self.group_columns[0]}_{value_feature}_max"]
            X_final[f"{value_feature}/{self.group_columns[0]}_min"] = X_final[f"{value_feature}"] / ((X_final[f"{self.group_columns[0]}_{value_feature}_min"]**2)+10)
        
        return X_final


"""
==============================================================================
Classe para selecionar as features finais que vão para o treinamento do modelo
==============================================================================

Retorna um pandas dataframe com as colunas originais e as novas features calculadas.

:: Parâmetros 
    ** features (requerido): list
        Lista com o nome das features que serão selecionadas e utilizadas para treinamento.

:: Exemplo:
-----------
    #Inicialização
    ff = FinalFeatures(features=["montante_pagamento_mes1", "montante_pagamento_mes2"])

    #fit
    ff.fit(df_train)

    #transform
    df_train = ff.transform(df_train)
    df_val = ff.transform(df_val)
    df_test = ff.transform(df_test)
"""
class FinalFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        data = data[self.features]

        return data