from roboflow import Roboflow
rf = Roboflow(api_key="sfGRWRbhP29Y2TV8i6PO")
project = rf.workspace("vrishank-umrani-s-workspace").project("ppe-gzzdx-nf8ps")
dataset = project.version(1).download("yolov8")

import os
with open("dataset_loc.txt", "w") as f:
    f.write(dataset.location)

print(f"Dataset downloaded successfully to: {dataset.location}")
print("Wrote dataset path to dataset_loc.txt")
