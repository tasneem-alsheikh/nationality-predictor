import torch
import os
import glob
import unicodedata
import string

MAX_LENGTH = 20

class NameDataset:
    def __init__(self, data_dir="names"):
        self.all_letters = string.ascii_letters + " -'"
        self.n_letters = len(self.all_letters)
        self.names = []
        self.nationalities = []
        self.nationality_to_ix = {}
        self.load_data(data_dir)

    def load_data(self, data_dir):
        """Load names from files and create indices."""
        file_pattern = os.path.join(data_dir, "*.txt")
        files = glob.glob(file_pattern)

        if not files:
            raise Exception(f"No .txt files found in {data_dir}")

        for filename in files:
            nationality = os.path.basename(filename).split('.')[0]
            if nationality not in self.nationality_to_ix:
                self.nationality_to_ix[nationality] = len(self.nationality_to_ix)

            with open(filename, encoding='utf-8') as f:
                for line in f:
                    name = self.unicode_to_ascii(line.strip())
                    if name:
                        self.names.append(name)
                        self.nationalities.append(self.nationality_to_ix[nationality])

    def unicode_to_ascii(self, s):
        """Convert Unicode string to ASCII."""
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in string.ascii_letters + " -'"
        )

    def name_to_tensor(self, name):
        """Convert name to tensor of character indices."""
        tensor = torch.zeros(MAX_LENGTH, dtype=torch.long)
        for i, char in enumerate(name[:MAX_LENGTH]):
            if char in self.all_letters:
                tensor[i] = self.all_letters.find(char)
        return tensor

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        nationality = self.nationalities[idx]
        return self.name_to_tensor(name), torch.tensor(nationality)
