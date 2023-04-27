import os
from utils.utils import ensure_dirs
import glob
from os.path import join, dirname, abspath
from datetime import datetime
import configargparse


class Config(object):
    def __init__(self, phase):
        self.is_train = phase == "train"

        # init hyperparameters and parse from command-line
        parser, args = self.parse()

        # set as attributes
        print("----Experiment Configuration-----")
        for k, v in sorted(args.__dict__.items()):
            self.__setattr__(k, v)
            print(f"{k:20}: {v}")

        # processing
        self.cont = self.cont_ckpt is not None

        if self.is_train:
            if self.debug:
                self.exp_name, self.date = 'debug', 'debug'
            elif self.cont:
                # continue training
                self.exp_name, self.para, self.date, self.ckpt = self.cont_ckpt.split(
                    '/')
            else:
                # new training
                self.exp_name = self.get_expname()
                self.para = f'{self.layers}_{self.segments}'
                self.date = datetime.now().strftime('%b%d_%H%M%S')
        else:
            # try:  # mostly used for load checkpoint
            #     self.exp_name, self.para, self.date, self.ckpt = self.test_ckpt.split(
            #     '/')
            # except: # used for only evaluate checkpoint
            self.ckpt_dir = dirname(self.test_ckpt)
            print(self.ckpt_dir)
            # GPU usage
            if args.gpu_ids is not None:
                print(f"my gpu id: {args.gpu_ids}")
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
                print(os.environ["CUDA_VISIBLE_DEVICES"])
            return 

        print(f'exp name: {self.exp_name}')

        # log folder
        proj_root = dirname(os.path.abspath(__file__))
        print(f'proj root: {proj_root}')

        self.log_dir = join(
            proj_root, f'exps_{self.dataset}', self.exp_name, self.para, self.date)
        self.model_dir = join(
            proj_root, f'exps_{self.dataset}', self.exp_name, self.para, self.date)
        #self.model_dir = join(proj_root, 'target_fn', self.target_fn, self.dist, f'{self.layers}_{self.segments}', self.permute, self.row_and_column, self.date)

        if not self.is_train or self.cont:
            assert os.path.exists(
                self.log_dir), f'Log dir {self.log_dir} does not exist'
            assert os.path.exists(
                self.model_dir), f'Model dir {self.model_dir} does not exist'
        else:
            ensure_dirs([self.log_dir, self.model_dir])

        if self.is_train:
            # save all the configurations and code
            log_name = f"log_cont_{datetime.now().strftime('%b%d_%H%M%S')}.txt" if self.cont else 'log.txt'
            py_list = sorted(
                glob.glob(join(dirname(abspath(__file__)), '**/*.py'), recursive=True))

            with open(join(self.log_dir, log_name), 'w') as log:
                for k, v in sorted(self.__dict__.items()):
                    log.write(f'{k:20}: {v}\n')
                log.write('\n\n')
                for py in py_list:
                    with open(py, 'r') as f_py:
                        log.write(f'\n*****{f_py.name}*****\n')
                        log.write(f_py.read())
                        log.write('================================================'
                                  '===============================================\n')

        # GPU usage
        if args.gpu_ids is not None:
            print(f"my gpu id: {args.gpu_ids}")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids)
            print(os.environ["CUDA_VISIBLE_DEVICES"])

    def parse(self):
        parser = configargparse.ArgumentParser(
            default_config_files=['settings/base.yml'])
        parser.add_argument('--config', is_config_file=True,
                            help='config file path')
        self._add_basic_config_(parser)
        self._add_flow_config_(parser)
        self._add_dataset_config_(parser)
        self._add_network_config_(parser)
        self._add_training_config_(parser)
        self._add_evaluation_config_(parser)

        if not self.is_train:
            self._add_test_config_(parser)
        args = parser.parse_args()
        return parser, args

    def _add_basic_config_(self, parser):
        group = parser.add_argument_group('basic')
        # what's the meaning?
        group.add_argument('--exp_name', type=self.str2type)
        group.add_argument('--suff_name', type=str,
                           help='name suffix appended after default exp_name')
        group.add_argument('--dist', type=str,
                           help='which distribution to use, typically mobiusflow')
        group.add_argument('--condition', type=int, default=0,
                           help='conditional task or not')
        group.add_argument('--RD_SEED', type=int,
                           default=42, help='random seed')
        group.add_argument('--category_num', type=int, default=1, help='number of categories to train at once')
        group.add_argument('--embedding', type=int, default=0, help='whether or not use category embedding')
        group.add_argument('--embedding_dim', type=int, default=32, help='embedding feature dimensions')
        # circularsplienflow, mobiusflow
        #group.add_argument('-prior', type = str, help = 'which prior distribution to use')
        # uniform, matrix_fisher
        return group

    def _add_flow_config_(self, parser):
        group = parser.add_argument_group('flow')
        group.add_argument('--pretrain_fisher', type=str,
                           default=None, help='path to pretrained fisher model')
        group.add_argument('--layers', type=int,
                           help='number of stacked layers')
        group.add_argument('--segments', type=int,
                           help='number of combination of transformation in each mobius flow')
        group.add_argument('--rot', type=str, default='16Trans',
                           help='type of affine layers, typically quaternion affine layers')
        group.add_argument('--lu', type=int, default=0, help='use lu parameterization for affine layers or not')
        return group

    def _add_dataset_config_(self, parser):
        group = parser.add_argument_group('dataset')
        group.add_argument('--data_dir', type=str, help='root for data')
        # group.add_argument('--sym', type=int, default=0)
        group.add_argument(
            '--dataset', type=str, choices=['modelnet', 'pascal3d', 'symsol', 'pose', 'raw'])
        group.add_argument('--category', type=str, help='specify which category in given dataset')
        group.add_argument('--length', type=int, default=0, help='specify dataset size, mostly not used')
        group.add_argument('--eval_category', type=str, help='specify which category to eval in training')
        return group

    def _add_network_config_(self, parser):
        """add hyperparameters for network architecture"""
        group = parser.add_argument_group('network')
        group.add_argument(
            "--network", type=str, choices=['mobilenet', 'resnet18', 'resnet50', 'resnet101'], help='network backbone')
        group.add_argument('--feature_dim', type=int,
                           default=32, help='feature dimensions')
        group.add_argument('--last_affine', type=int,
                           default=0, help='use affine transformation in the last layer of flows')
        group.add_argument('--first_affine', type=int,
                           default=1, help='use affine transformation in the first layer of flows')
        group.add_argument('--frequent_permute', type=int, default=0, help='can permute strategy in combining layers of mobius flows')
        return group

    def _add_training_config_(self, parser):
        group = parser.add_argument_group('training')
        group.add_argument('--lr', type=float, help="initial learning rate")
        group.add_argument('--batch_size', type=int, help="batch size")
        group.add_argument('--num_workers', type=int,
                           help="number of workers for data loading")
        group.add_argument('--max_iteration', type=int,
                           help="total number of iterations to train for supervised")
        group.add_argument('--log_frequency', type=int,
                           help="visualize output every x iterations")
        group.add_argument('--val_frequency', type=int,
                           help="run validation every x iterations")
        group.add_argument('--save_frequency', type=int,
                           help="save models every x iterations")
        group.add_argument('--cont_ckpt', type=str,
                           help="continue from checkpoint")
        group.add_argument('--number_queries', type=int,
                           help="number of points in evaluation")
        group.add_argument('-g', '--gpu_ids', type=str)
        group.add_argument('--debug', action='store_true',
                           help='debugging mode to avoid generating log files')
        
        group.add_argument('--use_lr_decay', type=int, default=1, help='whether to use leraning rate decay')
        group.add_argument('--lr_decay', type=str,
                           default="", help='learning rate decay strategy')
        group.add_argument('--gamma', type=float, default=1.0, help='used in learning rate decay')
        #group.add_argument('--freeze', type=str, default=None)

        
        return group

    def _add_evaluation_config_(self, parser):
        group = parser.add_argument_group('training')
        group.add_argument('--eval', type=str, help='evaluation metrics')
        group.add_argument('--scatter_size', type=float, default=1e-1, help='scatter size for points on the visualization plot')
        group.add_argument('--vis_idx', type=int, default=5, help='visualize which picture, used only in conditional tasks')
        group.add_argument('--eval_train', type=str, help='evaluation metrics used in training')
        return group

    def _add_test_config_(self, parser):
        group = parser.add_argument_group('test')
        group.add_argument('test_ckpt', type=str, help='checkpoint path')
        #group.add_argument('--hist_low', type=int, default=10)
        #group.add_argument('--hist_high', type=int, default=150)
        return group

    def get_expname(self):
        if self.exp_name is not None:
            exp_name = self.exp_name
        else:
            exp_name = f'{self.dist}_{self.category}_b{self.batch_size}_lr{self.lr:.1e}'
            if self.suff_name:
                exp_name += self.suff_name
        return exp_name

    @staticmethod
    def str2type(s):
        if str(s).lower() == 'true':
            return True
        elif str(s).lower() == 'false':
            return False
        elif str(s).lower() == 'none':
            return None
        else:
            return s


def get_config(phase):
    config = Config(phase)
    return config
