import pandas as pd

data_path = ""
data_frame = pd.read_csv(data_path)
for i in range(len(data_frame)):
    data_frame["Image"][i] = data_frame["Image"][i] + ".nrrd"

data_frame.to_csv(data_path, index=False)