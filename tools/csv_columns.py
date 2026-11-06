"""Read and write dataset CSVs as raw strings, one column per profile, so that
tools that select profiles copy every value exactly as written."""
import csv
import sys
from pathlib import Path

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


class ColumnTable:
    """Dataset CSV kept as strings: ``labels`` (row labels or None), ``ids``
    (profile ids) and ``rows`` (one list of cells per data row, without labels)."""

    def __init__(self, labels, ids, rows):
        self.labels, self.ids, self.rows = labels, list(ids), rows
        for row in rows:
            if len(row) != len(self.ids):
                raise ValueError("All rows must have one cell per profile")

    @classmethod
    def read(cls, path):
        with open(path, newline="") as handle:
            table = list(csv.reader(handle))
        if len(table) < 2:
            raise ValueError(f"{path}: not a dataset CSV")
        labelled = table[1] and table[1][0].strip().lower() == "year"
        for i, row in enumerate(table):
            if len(row) != len(table[0]):
                raise ValueError(f"{path}: row {i} has {len(row)} fields, header has {len(table[0])}")
        if labelled:
            return cls([row[0] for row in table[1:]], table[0][1:], [row[1:] for row in table[1:]])
        return cls(None, table[0], table[1:])

    def __len__(self):
        return len(self.ids)

    def select(self, indices):
        indices = list(indices)
        return ColumnTable(self.labels, [self.ids[i] for i in indices],
                           [[row[i] for i in indices] for row in self.rows])

    def years(self):
        return [float(v) for v in self.rows[0]]

    @staticmethod
    def concat(tables):
        labels = tables[0].labels
        for table in tables[1:]:
            if table.labels != labels or len(table.rows) != len(tables[0].rows):
                raise ValueError("Tables have different rows")
        ids = [i for table in tables for i in table.ids]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate profile ids across the input CSVs")
        rows = [[cell for table in tables for cell in table.rows[r]] for r in range(len(tables[0].rows))]
        return ColumnTable(labels, ids, rows)

    def write(self, path, overwrite=False):
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            if self.labels is None:
                writer.writerow(self.ids)
                writer.writerows(self.rows)
            else:
                writer.writerow([""] + self.ids)
                writer.writerows([label] + row for label, row in zip(self.labels, self.rows))
