from flask import Flask, render_template, request, jsonify
import time
import json
import numpy as np
from collections import defaultdict
from typing import Self
import pickle
import pandas as pd
import os

data_folder = "dataset"

app = Flask(__name__)

filter = [13, 18, 26]

# Route to serve the HTML page
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle keystroke data
@app.route("/keystrokes", methods=["POST"])
def keystrokes():
    data = request.json
    print("Keystroke data received:", data)  # For debugging/logging
    # Here you can save the data to a database or file for further analysis

    #features = extract_features(data)

    print("  ")

    #print("Extracted features:", features["key_keydown"])  # For debugging/logging

    return jsonify({"status": "success"})

minimum_profile_length_r: int = 5
minimum_profile_length_a: int = 5

# control log level
trace: bool = False

# create and read user profiles. Format of keystroke data is assumed to be a list of tuples (key, timestamp)
def keystrokes_to_digraphs(keystroke_array):
    digraphs = []
    i = 0
    while i < len(keystroke_array) - 1:
        print(keystroke_array[i][0] + "-" + keystroke_array[i + 1][0])
        digraphs.append(
            (
                str(keystroke_array[i][0]) + "-" + str(keystroke_array[i + 1][0]),
                np.round((keystroke_array[i + 1][1] - keystroke_array[i][1]), 5),
            )
        )
        i += 1
    return digraphs


def keystrokes_to_trigraphs(keystroke_array):
    trigraphs = []
    i = 0
    while i < len(keystroke_array) - 2:
        trigraphs.append(
            (
                str(keystroke_array[i][0])
                + "-"
                + str(keystroke_array[i + 1][0])
                + "-"
                + str(keystroke_array[i + 2][0]),
                np.round((keystroke_array[i + 2][1] - keystroke_array[i][1]), 5),
            )
        )
        i += 1
    return trigraphs


def keystrokes_to_fourgraphs(keystroke_array):
    fourgraphs = []
    i = 0
    while i < len(keystroke_array) - 3:
        fourgraphs.append(
            (
                str(keystroke_array[i][0])
                + "-"
                + str(keystroke_array[i + 1][0])
                + "-"
                + str(keystroke_array[i + 2][0])
                + "-"
                + str(keystroke_array[i + 3][0]),
                np.round((keystroke_array[i + 3][1] - keystroke_array[i][1]), 5),
            )
        )
        i += 1
    return fourgraphs


def calculate_mean_for_duplicates(ngraphs):
    cleaned_ngraphs = []
    processed_keys = []
    for key, time in ngraphs:
        if key not in processed_keys:
            duplicates = [e for e in ngraphs if e[0] == key]
            if len(duplicates) > 1:
                processed_keys.append(key)
                cleaned_ngraphs.append(
                    (key, np.round(np.mean([d[1] for d in duplicates]), 5))
                )
            else:
                processed_keys.append(key)
                cleaned_ngraphs.append((key, time))
    return cleaned_ngraphs



def create_user_profile(keystroke_sequence):
    digraphs = calculate_mean_for_duplicates(keystrokes_to_digraphs(keystroke_sequence))
    trigraphs = calculate_mean_for_duplicates(
        keystrokes_to_trigraphs(keystroke_sequence)
    )
    fourgraphs = calculate_mean_for_duplicates(
        keystrokes_to_fourgraphs(keystroke_sequence)
    )
    return digraphs, trigraphs, fourgraphs

def read_file(complete: pd.DataFrame, user: int, set: int) -> list[(str, int)]:
    key_codes = complete.loc[(complete["user"] == user) & (complete["set"] == set)][
        "key"
    ].to_list()
    timestamps = complete.loc[(complete["user"] == user) & (complete["set"] == set)][
        "timestamp"
    ].to_list()

    keystrokes = [(str(k), t) for (k, t) in zip(key_codes, timestamps)]

    return keystrokes


def read_user_data(complete):
    users = []

    for user in range(1, 75):
        tmp_keystrokes = []
        for set in range(1, 3):
            f = read_file(complete, user, set)
            tmp_keystrokes.append(f)
        users.append(tmp_keystrokes)
    return users


def get_user_profiles(user_data):
    user_profiles = []
    count = 0
    for u_data in user_data:
        digraphs = []
        trigraphs = []
        fourgraphs = []
        for sample in u_data:
            tmp_digraphs, tmp_trigraphs, tmp_fourgraphs = create_user_profile(sample)
            digraphs.append(dict(tmp_digraphs))
            trigraphs.append(dict(tmp_trigraphs))
            fourgraphs.append(dict(tmp_fourgraphs))

        user_profiles.append(
            {"digraphs": digraphs, "trigraphs": trigraphs, "fourgraphs": fourgraphs}
        )
        count += 1
    return user_profiles


def create_user_profiles(path_to_userdata, filename):
    user_data2 = read_user_data(pd.read_csv(path_to_userdata))
    user_profiles = get_user_profiles(user_data2)
    with open(filename, "wb") as fp:
        pickle.dump(user_profiles, fp)

def extract_features(keystrokes):
    """
    Extracts features from the raw keystroke data for recognition.
    Features include key press durations, flight times, and typing speed.
    """
    key_durations = {}  # Dictionary to store key press durations
    flight_times = []   # List to store flight times between key events
    total_keys = 0      # Counter for total keys typed
    valid_keys = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ") # whitelist of valid keys

    key_keydown = {}

    # Iterate through keystroke events to calculate durations and flight times
    last_keyup_time = None
    i = 0
    for event in keystrokes:
        if event['key'] not in valid_keys:
            i = i + 1
            continue
        start_time = keystrokes[i]['timestamp']
        break
    end_time = keystrokes[-1]['timestamp'] if keystrokes else 0

    for event in keystrokes:
        if event['key'] not in valid_keys:
            continue
        if event['type'] == 'keydown':
    
            # Record the time of keydown for duration calculation
            key_durations[event['key']] = key_durations.get(event['key'], [])
            key_durations[event['key']].append(event['timestamp'])
            key_keydown[event['timestamp']-start_time] = event['key']
        elif event['type'] == 'keyup':
            # Calculate key press duration
            if event['key'] in key_durations and key_durations[event['key']]:
                keydown_time = key_durations[event['key']].pop(0)
                key_durations[event['key']].append(event['timestamp'] - keydown_time)
            
            # Count valid keys
            if event['key'] in valid_keys:
                total_keys += 1
            
            # Calculate flight time if a previous keyup exists
            if last_keyup_time is not None:
                flight_times.append(event['timestamp'] - last_keyup_time)
            last_keyup_time = event['timestamp']

    # Average durations and flight times
    avg_durations = {key: sum(durations) / len(durations) for key, durations in key_durations.items() if durations}
    avg_flight_time = sum(flight_times) / len(flight_times) if flight_times else 0

    # Calculate typing speed (WPM)
    total_time_seconds = (end_time - start_time) / 1000  # Convert milliseconds to seconds
    total_time_minutes = total_time_seconds / 60        # Convert seconds to minutes
    wpm = (total_keys / 5) / total_time_minutes if total_time_minutes > 0 else 0

    return {
        "average_key_durations": avg_durations,
        "average_flight_time": avg_flight_time,
        "typing_speed_wpm": round(wpm, 2),
        "total_time_seconds": total_time_seconds,
        "total_keys": total_keys,
        "key_keydown": key_keydown
    }

# abstractions
class Sample:
    def __init__(
        self,
        digraphs: dict[str, float],
        trigraphs: dict[str, float],
        fourgraphs: dict[str, float],
    ):
        self.digraphs = digraphs
        self.trigraphs = trigraphs
        self.fourgraphs = fourgraphs

    def __str__(self):
        return f"digraphs: {self.digraphs} trigraphs: {self.trigraphs} fourgraphs: {self.fourgraphs}"

    def get_intersection(self, other: Self) -> Self:
        intersection_digraphs = self.digraphs.keys() & other.digraphs.keys()
        intersection_trigraphs = self.trigraphs.keys() & other.trigraphs.keys()
        intersection_fourgraphs = self.fourgraphs.keys() & other.fourgraphs.keys()

        s_digraphs = {
            k: v for k, v in self.digraphs.items() if k in intersection_digraphs
        }
        s_trigraphs = {
            k: v for k, v in self.trigraphs.items() if k in intersection_trigraphs
        }
        s_fourgraphs = {
            k: v for k, v in self.fourgraphs.items() if k in intersection_fourgraphs
        }

        return Sample(s_digraphs, s_trigraphs, s_fourgraphs)

    def get_digraphs(self) -> dict[str, float]:
        return self.digraphs

    def get_trigraphs(self) -> dict[str, float]:
        return self.trigraphs

    def get_fourgraphs(self) -> dict[str, float]:
        return self.fourgraphs


class UserProfile:
    def __init__(self, profile: dict[str, list[dict]]):

        assert (
            len(profile["digraphs"])
            == len(profile["trigraphs"])
            == len(profile["fourgraphs"])
        )

        self.digraphs = profile["digraphs"]
        self.trigraphs = profile["trigraphs"]
        self.fourgraphs = profile["fourgraphs"]
        self.m_cache = None

    def __str__(self):
        return f"digraphs: {self.digraphs} trigraphs: {self.trigraphs} fourgraphs: {self.fourgraphs}"

    def get_sample_count(self) -> int:
        return len(self.digraphs)

    def get_sample(self, index: int) -> Sample:
        return Sample(
            self.digraphs[index], self.trigraphs[index], self.fourgraphs[index]
        )

    def get_samples(self) -> list[Sample]:
        out = [self.get_sample(i) for i in range(self.get_sample_count())]
        return out
    
    def get_expected_keys(self) -> list[str]:
        return list(self.get_samples()[0].get_digraphs().keys())

    def m_without_x(self,x) -> dict[str, float]:
        if x > 14:
            print("sample" + str(x))

        samples: list[Sample] = self.get_samples()

        filtered_samples = [sample for i, sample in enumerate(samples) if i != x]

        distances: dict[str, list[float]] = defaultdict(list)

        # calculate distances from each set in profile
        for i, sample_A in enumerate(filtered_samples):
            for j, sample_B in enumerate(filtered_samples):
                # distance from same sample does not have to be calculated
                if j == i:
                    assert sample_A == sample_B
                    continue
                    
                # calculate distance between two samples
                distance_combinations: dict[str, float] = d(sample_A, sample_B)
        
                # append each distance to distances
                for key, value in distance_combinations.items():
                    distances[key].append(value)

        return {k: np.array(v).mean() for k, v in distances.items()}

    def m(self) -> dict[str, float]:
        # check if m was already calculated for this profile
        if self.m_cache is not None:
            return self.m_cache

        samples: list[Sample] = self.get_samples()

        distances: dict[str, list[float]] = defaultdict(list)

        # calculate distances from each set in profile
        for i, sample_A in enumerate(samples):
            for j, sample_B in enumerate(samples):
                # distance from same sample does not have to be calculated
                if j == i:
                    assert sample_A == sample_B
                    continue

                # calculate distance between two samples
                distance_combinations: dict[str, float] = d(sample_A, sample_B)

                # append each distance to distances
                for key, value in distance_combinations.items():
                    distances[key].append(value)

        # calculate mean for each distance and return
        self.m_cache = {k: np.array(v).mean() for k, v in distances.items()}
        return self.m_cache

# basic distances
def a_distance(
    sample_A_ngraphs: dict[str, float],
    sample_B_ngraphs: dict[str, float],
    threshold: float = 1.05,
) -> float:

    assert len(sample_A_ngraphs) == len(sample_B_ngraphs)

    # check that a minimal number of digraphs are shared
    number_of_shared_ngraphs = len(sample_A_ngraphs)
    if number_of_shared_ngraphs < minimum_profile_length_a:
        if trace:print(f"[TRACE]: Insufficient number of n-graphs: {number_of_shared_ngraphs}")
        return 1

    similar_ngraphs: int = 0

    # for each n-graph
    for n_graph in sample_A_ngraphs:

        d1: float = sample_A_ngraphs[n_graph]
        d2: float = sample_B_ngraphs[n_graph]

        # if distance for two inputs is 0,
        # set to very small number, to prevent division by 0
        # TODO: is this ok?
        if d1 == 0:
            d1 = 0.0000001

        if d2 == 0:
            d2 = 0.0000001

        # 1 < max(d1, d2)/min(d1, d2) ≤ t
        if 1 < max(d1, d2) / min(d1, d2) <= threshold:
            similar_ngraphs += 1

    distance: float = 1 - (similar_ngraphs / number_of_shared_ngraphs)

    return np.round(distance, 6)

def r_distance(
    sample_A_ngraphs: dict[str, float], sample_B_ngraphs: dict[str, float]
) -> float:
    assert len(sample_A_ngraphs) == len(sample_B_ngraphs)

    # check that a minimal number of digraphs are shared
    number_of_shared_ngraphs = len(sample_A_ngraphs)
    if number_of_shared_ngraphs < minimum_profile_length_a:
        if trace:print(f"[TRACE]: Insufficient number of n-graphs: {number_of_shared_ngraphs}")
        return 1

    # order reference(user profile) n-graphs based on n-grpah duration
    sample_A_ngraphs_sorted = list(dict(sorted(sample_A_ngraphs.items(), key=lambda item: item[1])))

    # order sample n-graphs based on n-grpah duration
    sample_B_ngraphs_sorted = list(dict(sorted(sample_B_ngraphs.items(), key=lambda item: item[1])))

    # calculate distances between n-graph positions in reference and evaluation datasets
    ordered_distances = [abs(sample_A_ngraphs_sorted.index(ele) - idx)for idx, ele in enumerate(sample_B_ngraphs_sorted)]

    # calculate maximum degree of disorder
    # (if |V| is even) 0> (|V|^2 / 2)
    if number_of_shared_ngraphs % 2 == 0:
        maximum_disorder = ((number_of_shared_ngraphs * number_of_shared_ngraphs)) / 2
    # (if |V| is odd) => (|V|^2 − 1) / 2
    else:
        maximum_disorder = (
            (number_of_shared_ngraphs * number_of_shared_ngraphs) - 1
        ) / 2

    # calculate r-distance
    distance = np.sum(ordered_distances) / maximum_disorder

    return np.round(distance, 6)

# generic d(distance) and md(mean distance) functions
def d(sample_A: Sample, sample_B: Sample) -> dict[str, float]:

    # get shared n-graphs
    shared_sample_A = sample_A.get_intersection(sample_B)
    shared_sample_B = sample_B.get_intersection(sample_A)

    assert (shared_sample_A.get_digraphs().keys() == shared_sample_B.get_digraphs().keys())
    assert (shared_sample_A.get_trigraphs().keys() == shared_sample_B.get_trigraphs().keys())
    assert (shared_sample_A.get_fourgraphs().keys()== shared_sample_B.get_fourgraphs().keys())

    # get basic distances
    a2 = a_distance(shared_sample_A.get_digraphs(), shared_sample_B.get_digraphs())
    a3 = a_distance(shared_sample_A.get_trigraphs(), shared_sample_B.get_trigraphs())
    a4 = a_distance(shared_sample_A.get_fourgraphs(), shared_sample_B.get_fourgraphs())

    r2 = r_distance(shared_sample_A.get_digraphs(), shared_sample_B.get_digraphs())
    r3 = r_distance(shared_sample_A.get_trigraphs(), shared_sample_B.get_trigraphs())
    r4 = r_distance(shared_sample_A.get_fourgraphs(), shared_sample_B.get_fourgraphs())

    # will contain all combinations of a- and r-distances
    out: dict[str, float] = {}

    out["a2"] = a2
    out["a3"] = a3
    out["a4"] = a4

    out["r2"] = r2
    out["r3"] = r3
    out["r4"] = r4

    out["a23"] = a2 + a3
    out["a34"] = a3 + a4
    out["a234"] = a2 + a3 + a4

    out["r23"] = r2 + r3
    out["r34"] = r3 + r4
    out["r234"] = r2 + r3 + r4

    out["r2_a2"] = r2 + a2
    out["r3_a3"] = r3 + a3
    out["r4_a4"] = r4 + a4

    out["r23_a23"] = r2 + r3 + a2 + a3
    out["r34_a34"] = r3 + r4 + a3 + a4
    out["r24_a24"] = r2 + r4 + a2 + a4

    out["r234_a234"] = r2 + r3 + r4 + a2 + a3 + a4

    out["r2_a3"] = r2 + a3
    out["r2_a4"] = r2 + a4

    out["r2_a24"] = r2 + a2 + a4

    out["r3_a2"] = r3 + a2
    out["r4_a2"] = r4 + a2

    out["r23_a2"] = r2 + r3 + a2
    out["r23_a3"] = r2 + r3 + a3
    out["r23_a4"] = r2 + r3 + a4

    out["r234_a2"] = r2 + r3 + r4 + a2
    out["r234_a3"] = r2 + r3 + r4 + a3
    out["r234_a4"] = r2 + r3 + r4 + a4

    out["r234_a23"] = r2 + r3 + r4 + a2 + a3

    out["r2_a234"] = r2 + a2 + a3 + a4
    out["r3_a234"] = r3 + a2 + a3 + a4
    out["r4_a234"] = r4 + a2 + a3 + a4

    out["r23_a234"] = r2 + r3 + a2 + a3 + a4
    out["r34_a234"] = r3 + r4 + a2 + a3 + a4

    return out


def md(user: UserProfile, sample: Sample, user_sample_skip: None | int = None) -> dict[str, float]:
    """
    Calculates the mean distances between the user profile and the sample.

            Parameters:
                    user (UserProfile): A user profile to calculate the distance to
                    sample (Sample): A sample to calculate the distance from
                    user_sample_skip (optional int): An index for a sample to skip

            Returns:
                    index (dict[str, float]): The mean distance combinations
    """
    assert isinstance(user, UserProfile), f"Wrong input type: {type(user)}"
    assert isinstance(sample, Sample), f"Wrong input type: {type(sample)}"

    distances: dict[str, list[float]] = defaultdict(list)

    # calculate distance to each set from user profile
    for i, user_sample in enumerate(user.get_samples()):
        # skip this sample
        if user_sample_skip is not None and user_sample_skip == i:
            continue

        distance_combinations: dict[str, float] = d(user_sample, sample)

        # append each distance to distances
        for key, value in distance_combinations.items():
            distances[key].append(value)

    # calculate mean for each distance and return
    return {k: np.array(v).mean() for k, v in distances.items()}


# user classification
def user_classification(distances: dict[str, list[float]], distance_measure: str) -> int:

    # will contains the r234_a23 distance for each user
    user_distances = distances[distance_measure]

    # returns the index(user) with the minimal distance
    return np.argmin(user_distances)

# user authentification
def authentification(mean_distances: dict[str, list[float]], measure: str, attacked_user: UserProfile, attacked_index: int) -> bool:
    # get the mean distance of the UserProfile
    m_A = attacked_user.m()[measure]

    # get the mean distance from UserProfile to sample
    md_A_X = mean_distances[measure][attacked_index]

    # check, that distance from UserProfile to sample is small enought
    for md_B_X in mean_distances[measure]:
        # md(A, X) < m(A) + 0.5[md(B,X) − m(A)]
        if not md_A_X < m_A + (0.5 * (md_B_X - m_A)):
            # some other UserProfile has small enought distance to sample, authentification fail
            return False
        
    # 
    return True

# user authentification for attacker sample belonging to attacked user
def authentification_legitimate(mean_distances: dict[str, list[float]], measure: str, attacked_user: UserProfile, attacked_index: int, sample_index: int) -> bool:
    # get the mean distance of the UserProfile
    if attacked_user.get_sample_count() <= 2:
        return False

    m_A = attacked_user.m_without_x(sample_index)[measure]

    # get the mean distance from UserProfile to sample
    md_A_X = mean_distances[measure][attacked_index]

    # check, that distance from UserProfile to sample is small enought
    for index, md_B_X in  enumerate(mean_distances[measure]):
        if index != attacked_index:
            # md(A, X) < m(A) + 0.5[md(B,X) − m(A)]
            if not md_A_X < m_A + (0.5 * (md_B_X - m_A)):
                # some other UserProfile has small enought distance to sample, authentification fail
                return False
        
    # 
    return True

# mean distances(md) calculation
def get_mean_distances(attacked_user_profiles: list[UserProfile], attacker_sample: Sample, attacker_sample_idx: int, attacker_index: int) -> dict[str, list[float]]:
    """
    Calculates the mean distances between each user profile and the sample.

            Parameters:
                    attacked_user_profiles (list[UserProfile]): List of UserProfile to calculte distance from
                    attacker_sample (Sample): A sample to calculate the distance to
                    attacker_sample_idx (int): Index of sample
                    attacker_index (int): Index of attacker Profile the Sample comes from 

            Returns:
                    index (dict[str, list[float]]): Returns the mean distance from each UserProfile to sample for each distance measure
    """

    # calculate distances from sample to user profiles
    distances: list[dict[str, float]] = []
    for attacked_index, user_profile in enumerate(attacked_user_profiles):
        # its the same user,
        # pass sample index to skip this sample
        if attacked_index == attacker_index:
            distance = md(user_profile, attacker_sample, attacker_sample_idx)
            distances.append(distance)
        # it not the same user, don't skip sample
        else:
            distance = md(user_profile, attacker_sample, None)
            distances.append(distance)


    # transform from list of dicts to dict of lists
    distances_converted: dict[str, list[float]] = defaultdict(list)
    for entry in distances:
        for key, value in entry.items():
            distances_converted[key].append(value)

    return distances_converted


# execute experiment
def execute(
    user_profiles_training: list[UserProfile],
    user_profiles_evaluation: list[UserProfile],
) -> dict[str, list[dict]]:
    # keys
    FRA = "FalseRejectAttempt"
    FAA = "FalseAcceptAttempt"
    FRS = "FalseRejectSucess"
    FAS = "FalseAcceptSucess"
    FRE = "FalseRejectError"
    FAE = "FalseAcceptError"
    FR1 = "FalseReject1"
    FR2 = "FalseReject2"
    
    out = {"class": [], "auth": []}

    auth_score: dict[str, dict[str, int]] = {}
    class_score: dict[str, dict[str, int]] = {}

    # for every user
    for attacker_index, attacker_user in enumerate(user_profiles_evaluation):
        print(f"progress: {((attacker_index + 1) / len(user_profiles_evaluation)) * 100:.1f}%", end='\r')
        # attack with each sample
        for attacker_sample_index, attacker_sample in enumerate(attacker_user.get_samples()):
            
            # calculate distances between attacker sample and each attacked user profiles
            mean_distances: dict[str, list[float]] = get_mean_distances(user_profiles_training, attacker_sample, attacker_sample_index, attacker_index)

            # every attacked user
            for attacked_index, attacked_user in enumerate(user_profiles_training):
                # try classification and authentification for each distance measure
                for distance_measure in mean_distances.keys():
                    # init counter
                    if class_score.get(distance_measure) is None: class_score[distance_measure] = {FRA:0,FAA:0,FRS:0,FAS:0,FRE:0,FAE:0, FR1:0, FR2:0}
                    if auth_score.get(distance_measure) is None: auth_score[distance_measure] = {FRA:0,FAA:0,FRS:0,FAS:0,FRE:0,FAE:0, FR1:0, FR2:0}

                    # try to classifiy user
                    # user with smallest distance from profile to sample
                    classified_user_index = user_classification(mean_distances, distance_measure)

                    # check if attacker and attacked are the same
                    same_user = attacker_index == attacked_index

                    # check if the attacked user was classified
                    attacked_user_classifified = classified_user_index == attacked_index

                    # valid classification/authentification -> False Reject Attempt
                    if same_user:
                        class_score[distance_measure][FRA] += 1
                        auth_score[distance_measure][FRA] += 1
                    # imposter classification/authentification -> False Accept Attempt
                    else:
                        class_score[distance_measure][FAA] += 1
                        auth_score[distance_measure][FAA] += 1
    
                    # user attacks itself and classification suceeded, 
                    # False Reject
                    if same_user and attacked_user_classifified:
                        # False Reject Success
                        class_score[distance_measure][FRS] += 1

                        # check second auth step
                        if authentification_legitimate(mean_distances, distance_measure, attacked_user, attacked_index,attacker_sample_index):
                        #if authentification(mean_distances, distance_measure, attacked_user, attacked_index):
                            # False Reject Sucess
                            auth_score[distance_measure][FRS] += 1
                        else:
                            # False Reject Error
                            auth_score[distance_measure][FRE] += 1
                            auth_score[distance_measure][FR1] += 1

                    # user attacks itself and classification failed
                    # False Reject Error
                    elif same_user and not attacked_user_classifified:
                        # False Reject Error
                        class_score[distance_measure][FRE] += 1
                        auth_score[distance_measure][FRE] += 1
                        auth_score[distance_measure][FR2] += 1
                
                    # user attacks other user and classification suceeded
                    # False Accept
                    elif not same_user and attacked_user_classifified:
                        # False Accept Error
                        class_score[distance_measure][FAE] += 1

                        # check second auth step
                        if authentification(mean_distances, distance_measure, attacked_user, attacked_index):
                            # False Accept Error
                            auth_score[distance_measure][FAE] += 1
                        else:
                            # False Accept Sucess
                            auth_score[distance_measure][FAS] += 1

                    # user attacks other user and classification failed
                    # False Accept Success
                    elif not same_user and not attacked_user_classifified:
                        class_score[distance_measure][FAS] += 1
                        auth_score[distance_measure][FAS] += 1

                    else: assert False

    # check results
    for distance_measure in class_score.keys():
        c_value: dict[str, int] = class_score[distance_measure]
        a_value: dict[str, int] = auth_score[distance_measure]

        # attempts = sucess + errors
        assert c_value[FRA] == (c_value[FRS] + c_value[FRE])
        assert c_value[FAA] == (c_value[FAS] + c_value[FAE])

        # attempts = sucess + errors
        assert a_value[FRA] == (a_value[FRS] + a_value[FRE])
        assert a_value[FAA] == (a_value[FAS] + a_value[FAE])

        # classification False Accept Error 
        # have to be equal or greater then authentifiation False Accept Errors
        # because the second auth check can correct a wrong classification
        assert c_value[FAE] >= a_value[FAE]

        # auth False Reject Error 
        # have to be equal or greater then classification False Reject Errors
        # because the second auth check can fail a successfull classification
        assert a_value[FRE] >= c_value[FRE]

    # produce output
    for (distance_measure, value) in class_score.items():
        out["class"].append({
            "dist": distance_measure, 
            "FalseAcceptAttempts" : value[FAA],
            "FalseRejectAttempts": value[FRA],
            #"FalseAcceptSucess": value[FAS],
            #"FalseRejectSucess": value[FRS],
            "FalseAcceptError": value[FAE],
            "FalseRejectError": value[FRE], 
            })

    for (distance_measure, value) in auth_score.items():
        out["auth"].append({
            "dist": distance_measure, 
            "FalseAcceptAttempts" : value[FAA],
            "FalseRejectAttempts": value[FRA],
            #"FalseAcceptSucess": value[FAS],
            #"FalseRejectSucess": value[FRS],
            "FalseAcceptError": value[FAE],
            "FalseRejectError": value[FRE], 
            "FalseReject1": value[FR1],
            "FalseReject2": value[FR2]
        })

    return out

# experiment setup and output
def experiment(path_to_dataset_training: str,path_to_dataset_evaluation: str, output: str, filter: list = []): 
    # open training data set
    with open(path_to_dataset_training, "rb") as fp:
        user_profiles_training = pickle.load(fp)

    # open eval data sets
    with open(path_to_dataset_evaluation, "rb") as fp:
        user_profiles_evaluation = pickle.load(fp)

    # remove filtered rows: [13, 18, 26]
    user_profiles_training = [UserProfile(j) for i, j in enumerate(user_profiles_training) if i not in filter]
    user_profiles_evaluation = [UserProfile(j) for i, j in enumerate(user_profiles_evaluation) if i not in filter]

    results: dict[str, list[dict]] = execute(user_profiles_training, user_profiles_evaluation)
    
    class_results: list[dict] = results["class"]
    auth_results: list[dict] = results["auth"]

    class_results_df: pd.DataFrame = pd.DataFrame(class_results)
    auth_results_df: pd.DataFrame = pd.DataFrame(auth_results)

    class_results_df.to_csv(f"./{data_folder}/" + output + "_classification_data.csv", index=False)
    auth_results_df.to_csv(f"./{data_folder}/" + output + "_authentification_data.csv", index=False)


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

@app.route("/experiment", methods=["POST"])
def execute_experiment():
    # original data
    extract_from_buffalo()
    original_set = "./dataset/keystroke_baseline_task1.csv"
    original_data_profiles = f"./{data_folder}/original_data_profiles"

    print("Original data profiles: ", original_data_profiles)

    if not os.path.isfile(original_data_profiles):
        create_user_profiles(original_set, original_data_profiles)

    experiment(original_data_profiles, original_data_profiles, "original", filter)
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True)