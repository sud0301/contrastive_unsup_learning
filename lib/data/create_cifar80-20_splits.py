import os
import json
import random
from datetime import datetime

data_root = "/misc/lmbraid19/galessos/datasets/"

id = datetime.now().strftime("%y%m%d%H%M%S")
classes = list(range(100))
random.shuffle(classes)

classes_cifar_80 = classes[:80]
classes_cifar_20 = classes[80:]
assert len(classes_cifar_20) == 20 and len(classes_cifar_80) == 80

classes = {"cifar80": classes_cifar_80, "cifar20": classes_cifar_20}

with open(os.path.join(data_root, "cifar80-20", "cifar80-20_set{}.json".format(id)), "w") as f:
    json.dump(classes, f)

print("Written cifar80-20_set{}.json".format(id))



