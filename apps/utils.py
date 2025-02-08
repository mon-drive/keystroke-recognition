import pandas as pd
import os
from apps.GunettiPicardi import create_user_profiles, experiment
import csv
from collections import defaultdict
import re
from flask import request
import numpy as np

data_folder = "dataset"

filter = [13, 18, 26]


def process_txt_file(file_path, user_id, session_id):
    """Legge un file txt e restituisce un DataFrame con i dati processati."""
    data = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:  # Verifica che la riga abbia 3 elementi
                key, event, timestamp = parts
                if event == "KeyDown":  # Considera solo gli eventi KeyDown
                    data.append({
                        "user": user_id,
                        "set": session_id,
                        "key": key,
                        "timestamp": int(timestamp)
                    })
    return pd.DataFrame(data)

def convert_txt_to_csv(base_path, output_csv, text_type):
    """Legge i file dalla struttura di cartelle e li converte in un unico file CSV."""
    all_data = []
    sessions = ["s0", "s1", "s2"]  # Le tre sessioni disponibili
    for session_id, session_folder in enumerate(sessions):
        session_path = os.path.join(base_path, session_folder, "baseline")  # Solo baseline
        for file_name in os.listdir(session_path):
            # Controlla che il file sia per task 1 e un utente baseline
            if file_name.endswith(".txt"):
                try:
                    user_id = int(file_name[:3])  # ID utente: i primi 3 caratteri
                    task_id = int(file_name[5])  # Task ID: il sesto carattere
                    if 1 <= user_id <= 75 and task_id == text_type:  # Solo utenti baseline e task 1
                        file_path = os.path.join(session_path, file_name)
                        df = process_txt_file(file_path, user_id, session_id+1)
                        all_data.append(df)
                except ValueError:
                    print(f"Nome file non valido: {file_name}")

    # Concatena tutti i dati in un unico DataFrame
    final_df = pd.concat(all_data, ignore_index=True)
    
    # Salva in formato CSV
    final_df.to_csv(output_csv, index=False)
    print(f"File CSV salvato in: {output_csv}")

def extract_from_dataset(dataset: str):
        # Percorso alla directory contenente le cartelle delle sessioni

    output_csv = "./dataset/keystroke_baseline_task1.csv"

    print("Extracting from dataset:", dataset)

    if dataset == "Buffalo Fixed Text":
        base_path = "./dataset"
        convert_txt_to_csv(base_path, output_csv, 0)
    elif dataset == "Buffalo Free Text":
        base_path = "./dataset"
        convert_txt_to_csv(base_path, output_csv, 1)
    elif dataset == "Aalto":
        base_path = "./dataset/Aalto/files"
        processAaltoGP(base_path, output_csv, 1, 2000)
    else: #nanglae
        print("Dataset non riconosciuto")
        xls1 = "dataset/fullname_userInformation.xlsx"
        xls2 = "dataset/email_userInformation.xlsx"
        xls3 = "dataset/phone_userInformation.xlsx"
        convert_xlsx_to_csvGP([xls1,xls2,xls3], output_csv)
    

def execute_experimentGP(dataset: str):
    # original data
    print("Dataset dentro execute_experimentGP:", dataset)  
    extract_from_dataset(dataset)

    original_set = "./dataset/keystroke_baseline_task1.csv"
    original_data_profiles = f"./{data_folder}/original_data_profiles"

    print("Original data profiles: ", original_data_profiles)

    create_user_profiles(original_set, original_data_profiles)

    experiment(original_data_profiles, original_data_profiles, "original", filter)

def process_buffalo_keystrokes(input_path: str, output_csv: str, text_type):
    """
    Extracts keystroke dynamics features (H, UD, DD) from Buffalo dataset for task=0 (Fixed Text).
    
    Parameters:
        input_path (str): Path to the dataset directory.
        output_csv (str): Path to save the processed keystroke data.
    """
    data = []  # Store processed keystroke data
    repetitions = defaultdict(int)  # Count repetitions per user

    if text_type == 0:
        print("Processing keystroke data for task=0 (Fixed Text)")
    else:
        print("Processing keystroke data for task=1 (Free Text)")

    # Navigate through dataset folders and files
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".txt") and len(file) >= 6:
                user_id = file[:3]
                session = file[3]
                keyboard_type = file[4]
                task = file[5]

                # Process only task=0 (Fixed Text) or task=1 (Free Text)
                if task == str(text_type):
                    file_path = os.path.join(root, file)

                    # Read the file
                    with open(file_path, "r") as f:
                        lines = f.readlines()

                    # Parse key events
                    events = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            key, event_type, timestamp = parts
                            events.append({"key": key, "event_type": event_type, "timestamp": int(timestamp)})

                    # Compute H, DD, UD and count repetitions
                    hold_times = {}
                    for i in range(len(events) - 1):
                        current = events[i]
                        next_event = events[i + 1]

                        if current["event_type"] == "KeyDown" and next_event["event_type"] == "KeyUp" and current["key"] == next_event["key"]:
                            # Compute Hold Time (H.<key>)
                            hold_time = (next_event["timestamp"] - current["timestamp"]) / 1000.0  
                            hold_times[current["key"]] = hold_time

                        if current["event_type"] == "KeyUp" and next_event["event_type"] == "KeyDown":
                            # Compute Dwell Time (DD.<key1>.<key2>)
                            dwell_time = (next_event["timestamp"] - current["timestamp"]) / 1000.0  

                            # Compute Up-Down Time (UD.<key1>.<key2>)
                            up_down_time = (
                                (next_event["timestamp"] - events[i - 1]["timestamp"]) / 1000.0
                                if i > 0 else None
                            )

                            # Increase repetition count
                            repetitions[user_id] += 1

                            # Store data if hold time is available
                            if hold_times.get(current["key"], None) is None:
                                continue
                            data.append({
                                "subject": user_id,
                                "key": current["key"],
                                "H": hold_times.get(current["key"], None),
                                "UD": up_down_time,
                                "DD": dwell_time
                            })

    # Write data to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["subject", "key", "H", "UD", "DD"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def convert_xlsx_to_csv(input_files, output_file):
    dataframes = []
    
    for file in input_files:
        df = pd.read_excel(file, engine='openpyxl')
        df_converted = pd.DataFrame({
            'subject': df['id'],
            'key': df['single_alphabet'],
            'H': df['Dwell_time'] / 1000,         # Dwell time in secondi
            'UD': df['Flight_time'] / 1000,       # Flight time (Up-Down) in secondi
            'DD': df['Interval_time'] / 1000      # Interval time (Down-Down) in secondi
        })
        dataframes.append(df_converted)
    
    # Concatena tutti i DataFrame
    final_df = pd.concat(dataframes, ignore_index=True)
    
    # Salva il CSV
    final_df.to_csv(output_file, index=False)
    print(f"File salvato come {output_file}")

def processAalto(folder, output, min_user, max_user):
    all_data = []
    allowed_keys = re.compile(r'^[A-Za-z0-9]|SHIFT|CAPS_LOCK|CTRL|[.,-]$')
    for file in os.listdir(folder):
        if file.endswith("_keystrokes.txt"):
            user_id = int(file.split("_")[0])
            if user_id == 888:
                continue
            if min_user <= user_id <= max_user:
                file_path = os.path.join(folder, file)
                try:
                    df = pd.read_csv(file_path, sep='\t', dtype=str, on_bad_lines='skip', names=[
                        'PARTICIPANT_ID', 'TEST_SECTION_ID', 'SENTENCE', 'USER_INPUT', 
                        'KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'KEYCODE'
                    ])

                    expected_columns = {'PARTICIPANT_ID', 'TEST_SECTION_ID', 'SENTENCE', 'USER_INPUT', 
                    'KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'KEYCODE'}
                    if not expected_columns.issubset(df.columns):
                        raise ValueError(f"Il file {file} non contiene tutte le colonne necessarie: {df.columns}")

                                        
                    # Converti i dati numerici
                    df[['PRESS_TIME', 'RELEASE_TIME']] = df[['PRESS_TIME', 'RELEASE_TIME']].apply(pd.to_numeric, errors='coerce')
                    
                    # Rimuovi righe con valori NaN
                    df.dropna(inplace=True)

                    df.dropna(subset=['PRESS_TIME', 'RELEASE_TIME'], inplace=True)

                    df = df[(df['PRESS_TIME'] > 0) & (df['RELEASE_TIME'] > 0)]
                    
                    # Calcola H (Hold time)
                    df['H'] = (df['RELEASE_TIME'] - df['PRESS_TIME']) / 1000
                    
                    # Calcola UD (Up-Down time) e DD (Down-Down time)
                    df['UD'] = df['PRESS_TIME'].diff() / 1000
                    df['DD'] = df['PRESS_TIME'].diff().shift(-1) / 1000
                    df = df[(df['H'] >= 0) & (df['UD'] >= 0) & (df['DD'] >= 0)]
                    # Aggiungi subject e key
                    df['subject'] = user_id
                    df['key'] = df['LETTER'].fillna(df['KEYCODE'])

                    # Seleziona le colonne richieste
                    df = df[['subject', 'key', 'H', 'UD', 'DD']].dropna()

                    all_data.append(df)
                except Exception as e:
                    print(f"Errore nella lettura del file {file}: {e}")
    
    if all_data:
        # Unisce tutti i dati e salva il CSV
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output, index=False)
        print(f"File CSV salvato con successo in: {output}")
    else:
        print("Nessun dato valido trovato nei file.")

def processAaltoGP(folder, output, min_user, max_user):
    all_data = []
    for file in os.listdir(folder):
        if file.endswith("_keystrokes.txt"):
            user_id = int(file.split("_")[0])
            if user_id == 888:
                continue
            if min_user <= user_id <= max_user:
                file_path = os.path.join(folder, file)
                try:
                    df = pd.read_csv(file_path, sep='\t', dtype=str, on_bad_lines='skip', names=[
                        'PARTICIPANT_ID', 'TEST_SECTION_ID', 'SENTENCE', 'USER_INPUT', 
                        'KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'KEYCODE'
                    ])

                    expected_columns = {'PARTICIPANT_ID', 'TEST_SECTION_ID', 'SENTENCE', 'USER_INPUT', 
                    'KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'KEYCODE'}
                    if not expected_columns.issubset(df.columns):
                        raise ValueError(f"Il file {file} non contiene tutte le colonne necessarie: {df.columns}")

                                        
                    # Converti i dati numerici
                    df[['PRESS_TIME', 'RELEASE_TIME']] = df[['PRESS_TIME', 'RELEASE_TIME']].apply(pd.to_numeric, errors='coerce')
                    
                    # Rimuovi righe con valori NaN
                    df.dropna(inplace=True)

                    df.dropna(subset=['PRESS_TIME', 'RELEASE_TIME'], inplace=True)

                    df = df[(df['PRESS_TIME'] > 0) & (df['RELEASE_TIME'] > 0)]
                    
                    # Aggiungi subject e key
                    df['user'] = user_id
                    df['key'] = df['LETTER'].fillna(df['KEYCODE'])
                    df['set'] = 1
                    
                    df['timestamp'] = df['RELEASE_TIME']

                    # Seleziona le colonne richieste
                    df = df[['user', 'key', 'set', 'timestamp']].dropna()

                    all_data.append(df)
                except Exception as e:
                    print(f"Errore nella lettura del file {file}: {e}")
    
    if all_data:
        # Unisce tutti i dati e salva il CSV
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output, index=False)
        print(f"File CSV salvato con successo in: {output}")
    else:
        print("Nessun dato valido trovato nei file.")

def convert_xlsx_to_csvGP(input_files, output_file):
    # Lista per salvare i dati trasformati
    data_list = []

    for file in input_files:
        print(f"Processing {file}...")
        df = pd.read_excel(file)

        # Controlla che il file contenga le colonne necessarie
        required_columns = {"id", "single_alphabet", "Dwell_time"}
        if not required_columns.issubset(df.columns):
            print(f"Skipping {file}: Missing required columns")
            continue

        # Creazione delle nuove colonne
        df["user"] = df["id"]
        df["key"] = df["single_alphabet"]
        
        # Generiamo un valore per 'set' (ad esempio, il nome del file senza estensione)
        if file == "dataset/fullname_userInformation.xlsx":
            df["set"] = 1
        elif file == "dataset/email_userInformation.xlsx":
            df["set"] = 2
        else:
            df["set"] = 3

        # Creazione del timestamp progressivo
        base_time = 1471948497404  # Timestamp iniziale in millisecondi
        df["timestamp"] = base_time + df["Dwell_time"].cumsum()

        # Selezioniamo solo le colonne richieste
        df_final = df[["user", "key", "set", "timestamp"]]

        # Aggiungiamo alla lista
        data_list.append(df_final)

    # Uniamo tutti i dati in un DataFrame unico
    final_df = pd.concat(data_list, ignore_index=True)

    # Salviamo il CSV
    final_df.to_csv(output_file, index=False)

    print(f"CSV generato con successo: {output_file}")
