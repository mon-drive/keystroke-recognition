import pandas as pd
import os
from apps.GunettiPicardi import create_user_profiles, experiment
import csv
from collections import defaultdict

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

def convert_txt_to_csv(base_path, output_csv):
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
                    if 1 <= user_id <= 75 and task_id == 1:  # Solo utenti baseline e task 1
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

def extract_from_buffalo():
        # Percorso alla directory contenente le cartelle delle sessioni
    base_path = "./dataset"  # Sostituisci con il percorso reale
    output_csv = "./dataset/keystroke_baseline_task1.csv"

    convert_txt_to_csv(base_path, output_csv)

def execute_experimentGP():
    # original data
    extract_from_buffalo()
    original_set = "./dataset/keystroke_baseline_task1.csv"
    original_data_profiles = f"./{data_folder}/original_data_profiles"

    print("Original data profiles: ", original_data_profiles)

    if not os.path.isfile(original_data_profiles):
        create_user_profiles(original_set, original_data_profiles)

    experiment(original_data_profiles, original_data_profiles, "original", filter)

def process_keystrokes_with_repetitionsManhattan(input_path: str, output_csv: str):
    data = []  # Lista per raccogliere i dati da tutti i file
    repetitions = defaultdict(int)  # Conta le ripetizioni per ogni utente

    # Naviga nella cartella e sottocartelle
    for root, dirs, files in os.walk(input_path):
        for file in files:
            # Verifica che sia un file txt con il formato richiesto
            if file.endswith(".txt") and len(file) >= 6:
                user_id = file[:3]
                session = file[3]
                keyboard_type = file[4]
                task = file[5]

                # Filtra i file con task=1
                if task == "1":
                    file_path = os.path.join(root, file)

                    # Elabora il file
                    with open(file_path, "r") as f:
                        lines = f.readlines()

                    # Parsing dei dati
                    events = []
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            key, event_type, timestamp = parts
                            events.append({"key": key, "event_type": event_type, "timestamp": int(timestamp)})

                    # Calcolo H, DD, UD e aggiunta delle ripetizioni
                    hold_times = {}
                    for i in range(len(events) - 1):
                        current = events[i]
                        next_event = events[i + 1]

                        if current["event_type"] == "KeyDown" and next_event["event_type"] == "KeyUp" and current["key"] == next_event["key"]:
                            # Calcolo del hold time H.<key>
                            hold_time = (next_event["timestamp"] - current["timestamp"]) / 1000.0  # Decimale
                            hold_times[current["key"]] = hold_time

                        if current["event_type"] == "KeyUp" and next_event["event_type"] == "KeyDown":
                            # Calcolo del dwell time DD.<key1>.<key2>
                            dwell_time = (next_event["timestamp"] - current["timestamp"]) / 1000.0  # Decimale

                            # Calcolo dell'up-down time UD.<key1>.<key2>
                            up_down_time = (
                                (next_event["timestamp"] - events[i - 1]["timestamp"]) / 1000.0
                                if i > 0 else None
                            )

                            # Incrementa il contatore per il tasto corrente
                            repetitions[user_id] += 1

                            # Aggiungi i risultati
                            if hold_times.get(current["key"], None) is None:
                                continue
                            data.append({
                                "subject": user_id,
                                "key": current["key"],
                                "H": hold_times.get(current["key"], None),
                                "UD": up_down_time,
                                "DD": dwell_time
                            })

    # Scrivi il file CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            "subject", "key", "H", "UD", "DD"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)