# @Time   : 2022/3/22
# @Author : Jingsen Zhang
# @Email  : zhangjingsen@ruc.edu.cn

import argparse

from recbole_debias.quick_start.quick_inference import run_recbole_debias

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='REL_MF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Yahoo', help='name of datasets')
    parser.add_argument('--config_files', '-c', type=str, default=None, help='config files')
    parser.add_argument('--model_file', type=str, default=None, help='saved model path')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_debias(model=args.model, dataset=args.dataset, config_file_list=config_file_list, model_file=args.model_file)
