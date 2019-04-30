import os
import csv


class FeatureSet:
    def __init__(self, set_id, feature_set):
        self.set_id = set_id
        self.feature_set = feature_set

    def save_feature_set(self, settings_path):
        file_exists = os.path.isfile(settings_path)
        set_exists = False
        given_sets = []
        if file_exists:
            # Read already saved feature sets and their ID
            with open(settings_path, mode='r') as settings_file:
                i = 0
                settings_reader = csv.DictReader(settings_file, fieldnames=['ID', 'Features'], )
                for row in settings_reader:
                    if i > 0:
                        given_sets.append(row['Features'].split(','))
                    i += 1

        with open(settings_path, mode='a') as settings_file:
            settings_writer = csv.DictWriter(settings_file, delimiter=',', lineterminator='\n',
                                             fieldnames=['ID', 'Features'])
            if not file_exists:
                settings_writer.writeheader()
            else:
                for set in given_sets:
                    if sorted(set) == sorted(self.feature_set):
                        set_exists = True
                        break
            if not file_exists or not set_exists:
                row = {'ID': self.set_id, 'Features': ','.join(self.feature_set)}
                settings_writer.writerow(row)
