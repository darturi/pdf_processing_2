import os.path
import pickle

from build_database import QueryBase

key = "#"

qb = QueryBase(key)

qb.create_database(save_local=True)

qb_file_path = os.path.join("compute_resources", "qb.pkl")
f = open(qb_file_path, "wb")
pickle.dump(qb, f)
f.close()