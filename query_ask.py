import os.path
import pickle
from build_database import QueryBase

qb_file_path = os.path.join("compute_resources", "qb.pkl")
qb = None

if os.path.isfile(qb_file_path):
    f = open(qb_file_path, "rb")
    qb = pickle.load(f)
    f.close()
