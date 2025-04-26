import argparse
import time
from recbole.utils.utils import ensure_dir
from recbole.utils.utils import dict2str
from recbole.quick_start import run_recbole



def save_result_txt(run_result, result_path_prefix, dataset):
    idrd_flag = 1 if args.disen_lambda > 0 else 0
    result_file_path = result_path_prefix + '/' + str(args.model) + '_idrd' + str(idrd_flag) + '_idra' + str(args.idra)
    with open(result_file_path + '.txt', 'a+') as f:
        f.write('K: ' + str(args.n_newc_repr) + '\n')              #  K
        f.write('lambda_1: ' + str(args.disen_lambda) + '\n')      #  lambda_1
        f.write('lambda_2: ' + str(args.align_lambda) + '\n')      #  lambda_2

        f.write('model: ' + str(run_result['model']) + '\n')
        f.write('valid result:' + str(run_result['best_valid_result']) + '\n')

        over_metr = run_result['test_result']
        split_test_statis = run_result['split_test_statis']
        split_part_num = len(split_test_statis)
        for i in range(split_part_num):
            res = run_result['drift degree-{}_res'.format(str(i))]
            if res is not None:
                f.write('drift degree-{} res: {} \n \n'.format(str(i), res))

        f.write('## test result:' + str(run_result['test_result']) + '\n')
        f.write('\n')


if __name__ == '__main__':
    begin = time.time()
    parameter_dict = {
        'neg_sampling': None
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SASRec_IDURL', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='configs/Amazon_Beauty.yaml', help='config files')
    parser.add_argument('--rm_dup_inter', type=str, default=None)

    parser.add_argument('--split_eval', type=int, default=0)  # divide the user set into groups according to drift degree
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_heads', type=int, default=8)

    parser.add_argument('--n_newc_repr', type=int, default=4)       # K
    parser.add_argument('--disen_lambda', type=float, default=0.1)  # the weight of IDRD loss term
    parser.add_argument('--idra', type=int, default=1)              # add IDRA module
    parser.add_argument('--align_lambda', type=float, default=0.1)  # the weight of IDRA loss term

    args, _ = parser.parse_known_args()

    result_path_prefix = "run_results/" + str(args.dataset) + "/" + str(args.model)
    ensure_dir(result_path_prefix)

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_result = run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)
    end=time.time()
    print(end-begin)

    save_result_txt(run_result=run_result, result_path_prefix="run_results/" + str(args.dataset), dataset=args.dataset)
