import json
import sqlite3

import numpy as np
import pandas as pd

"""
Por algum motivo que eu desconheço,
quem criou essa matriz utilizou os códigos IBGE dos municípios
com 6 dígitos, sendo que eles têm 7 dígitos.

No entanto, se retirarmos o último dígito dos códigos com 7 dígitos,
continuamos com 5570 códigos distintos.
"""

def validar_codigos_ibge(lista_codigos_ibge):
    with open("data/raw/distance_matrix/municipios.json", "r", encoding="utf-8") as f:
        municipios = json.load(f)

    codigos_existentes = [mun["id"] for mun in municipios]
    codigos_existentes_reduzidos = [cod//10 for cod in codigos_existentes]

    lista_codigos_ibge_nova = list()
    for cod_ibge in lista_codigos_ibge:
        try:
            cod_ibge = int(cod_ibge)
        except:
            raise Exception(f"Código inválido: {cod_ibge}")
        
        if len(str(cod_ibge)) == 7 and cod_ibge in codigos_existentes:
            lista_codigos_ibge_nova.append(cod_ibge//10)
        
        elif len(str(cod_ibge)) == 6 and cod_ibge in codigos_existentes_reduzidos:
            lista_codigos_ibge_nova.append(cod_ibge)
        
        else:
            raise Exception(f"Código inexistente: {cod_ibge}")
    
    return lista_codigos_ibge_nova
        


def get_query(lista_codigos_ibge):
    query = f"""
    SELECT origem, destino, distancia, tempo
    FROM matriz_distancias
    WHERE (
        matriz_distancias.origem IN {tuple(lista_codigos_ibge)}
        AND
        matriz_distancias.destino IN {tuple(lista_codigos_ibge)}
    );
    """
    return query



def obter_df(lista_codigos_ibge):
    # retorna uma matriz do tipo dataframe
    # as colunas do dataframe são: origem, destino, distancia, tempo
    # origem e destino contém o código-IBGE reduzido (6 dígitos) do município
    lista_codigos_ibge_nova = validar_codigos_ibge(lista_codigos_ibge)

    con = sqlite3.connect("matriz_distancias.db")
    cur = con.cursor()

    query = get_query(lista_codigos_ibge_nova)
    response = cur.execute(query)
    result = response.fetchall()

    df_cols = ["origem", "destino", "distancia", "tempo"]
    df =  pd.DataFrame(result, columns=df_cols)
    n_rows = df.shape[0]
    if n_rows != len(lista_codigos_ibge_nova)**2:
        raise Exception(f"Encontradas {n_rows} distâncias. Número esperado: {len(lista_codigos_ibge)**2}")
    return df



def corrigir_codigo_ibge(cod_ibge:int, codigos_existentes:list):
    # cod_ibge deve ser um inteiro com 6 dígitos
    # essa é a forma que deveria ser encontrada na base de dados
    codigos_existentes_reduzidos = [cod//10 for cod in codigos_existentes]
    
    if cod_ibge not in codigos_existentes_reduzidos:
        raise Exception(f"Código reduzido inválido: {cod_ibge}")
    
    cod_idx = codigos_existentes_reduzidos.index(cod_ibge)
    return codigos_existentes[cod_idx]



def obter_matriz(lista_codigos_ibge):
    df = obter_df(lista_codigos_ibge)

    with open("data/raw/distance_matrix/municipios.json", "r", encoding="utf-8") as f:
        municipios = json.load(f)

    codigos_existentes = [mun["id"] for mun in municipios]
    
    df["origem"] = df["origem"].apply(lambda x: corrigir_codigo_ibge(x, codigos_existentes))
    df["destino"] = df["destino"].apply(lambda x: corrigir_codigo_ibge(x, codigos_existentes))

    matriz = list()
    matriz_cols = ["origem"] + lista_codigos_ibge 
    
    for origem in lista_codigos_ibge:
        row = [origem]
        for col_name in matriz_cols:
            if col_name != "origem":  # já considerado acima
                destino = col_name
                distancia = df.loc[(df["origem"] == origem) & (df["destino"] == destino)]["distancia"]
                distancia = distancia.tolist()[0]
                row.append(distancia)

        matriz.append(row)
    
    return pd.DataFrame(matriz, columns=matriz_cols)
