import pandas as pd
import os

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