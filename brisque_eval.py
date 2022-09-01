import os
from brisque import BRISQUE

def eval_brisque(path, status):
    files = os.listdir(path)

    all_brisque = 0
    idx = 0
    for i, f in enumerate(files):
        if status != None and status not in f:
            continue

        sub_path = os.path.join(path, f)
        brisque = BRISQUE(sub_path, url=False).score()
        all_brisque += brisque
        idx += 1
        print("{}/{}, {}:{}".format(i, len(files), sub_path, brisque))
    return all_brisque / (idx + 1)



if __name__ == "__main__":
    path = "/data/diffusion_data/infer/infer_128/hr_save/"
    status = '.'
    brisque = eval_brisque(path, status)
    print("Ave: {}".format(brisque))
