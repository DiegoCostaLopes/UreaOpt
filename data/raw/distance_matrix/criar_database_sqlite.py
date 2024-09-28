import sqlite3
import pandas as pd
import os

# matriz_distancias.csv deve ser baixado de:
# https://www.dropbox.com/scl/fo/zseuu3bmzsc9rj8ycpujd/AC9e1MUg0LpXaakjgsen3Do?rlkey=pz8htz0nlzdcd5q01n143k2en&e=1&dl=0

def main():
    df = pd.read_csv("matriz_distancias.csv")
    df = df[["origem", "destino", "distancia", "tempo"]]

    con = sqlite3.connect("matriz_distancias.db")
    cur = con.cursor()

    query1 = """
        CREATE TABLE IF NOT EXISTS matriz_distancias (
            origem INTEGER,
            destino INTEGER,
            distancia,
            tempo
        );"""
    
    cur.execute(query1)

    cur.executemany(
        "INSERT INTO matriz_distancias VALUES(?,?,?,?)",
        df.to_numpy()
    )
    con.commit()


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'matriz_distancias.db')
    
    if not os.path.exists(file_path):
        main()
        print("Pronto!")
