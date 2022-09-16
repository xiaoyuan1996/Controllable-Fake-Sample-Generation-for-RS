import os
from brisque import BRISQUE
import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str,
                        default= '/data/diffusion_data/infer/new_infer_512_256_220916_085236/results/sr_save/',
                        help='paths to images')
    args = parser.parse_args()
    status = 'sr.'
    brisque = eval_brisque(args.path, status)
    print("Ave: {}".format(brisque))
