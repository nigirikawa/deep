import glob
import re
import csv

files = glob.glob("test120190629*/dir*/*.hdf5")
pattern = ".*dir.*weights.(\d+)-(\d.\d+)-(\d.\d+)-(\d.\d+)-(\d.\d+).*"
with open("./result.csv", "w") as f:
    writer = csv.writer(f, lineterminator="\n")

    for file in files:
        result = re.match(pattern, file)
        if result:
            val_acc = float(result.group(5))
            writer.writerow([file, result.group(1), result.group(2), result.group(3), result.group(4), result.group(5)])

print("完了")