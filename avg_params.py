import torch
import argparse
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path1', '--path1', type=str, default="/data/diffusion_data/experiments/new_leader_221125_151235/lead_checkpoint/I1280000_E3627_gen.pth")
    parser.add_argument('-path2', '--path2', type=str, default="/data/diffusion_data/experiments/new_leader_221125_151235/lead_checkpoint/I1290000_E3655_gen.pth")
    parser.add_argument('-path3', '--path3', type=str, default="/data/diffusion_data/experiments/new_leader_221125_151235/lead_checkpoint/I1300000_E3683_gen.pth")
    parser.add_argument('-path4', '--path4', type=str,
                        default="/data/diffusion_data/experiments/new_leader_221125_151235/lead_checkpoint/I1310000_E3712_gen.pth")
    parser.add_argument('-path5', '--path5', type=str,
                        default="/data/diffusion_data/experiments/new_leader_221125_151235/lead_checkpoint/I1320000_E3740_gen.pth")
    parser.add_argument('-save_dir', '--save_dir', type=str, default="/data/diffusion_data/save_data/new_lead_checkpoint/")
    args = parser.parse_args()
    new_state ={'epoch': 1, 'iter': 1000000,
                     'scheduler': None, 'optimizer': None}
    if args.path1 is not None:
        opt1 = torch.load(args.path1)
        #state_dict1 = opt1['optimizer']
    if args.path2 is not None:
        opt2 = torch.load(args.path2)
        #state_dict2 = opt1['optimizer']
    if args.path3 is not None:
        opt3 = torch.load(args.path3)
    if args.path4 is not None:
        opt4 = torch.load(args.path4)
    if args.path5 is not None:
        opt5 = torch.load(args.path5)
        #state_dict3 = opt1['optimizer']
    new_dict = opt1
    for key,value in opt1.items():
        print(key)
        print(opt3[key])
        print(opt3[key])
        new_dict[key] = (opt1[key] + opt2[key] + opt3[key] + opt4[key] + opt5[key])/5
        print(value)
    #new_state['optimizer'] = new_dict
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir,'new_gen.pth')
    torch.save(new_dict,save_path)



