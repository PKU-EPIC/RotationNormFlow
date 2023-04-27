import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
import utils.utils as utils
import utils.sd as sd
from utils.networks import get_network
from flow.flow import get_flow
from utils.fisher import MatrixFisherN


def get_agent(config):
    return Agent(config)


class Agent:
    def __init__(self, config):
        self.config = config
        self.flow = get_flow(config)
        self.flow = DataParallel(self.flow)
        self.optimizer_flow = optim.Adam(self.flow.parameters(), config.lr)

        lr_decay = [int(item) for item in config.lr_decay.split(',')]
        scheduler_list = []
        scheduler_list.append(optim.lr_scheduler.MultiStepLR(
            self.optimizer_flow, milestones=lr_decay, gamma=config.gamma))

        if config.condition:
            self.net = get_network(config)
            optimizer_net_list = [*self.net.parameters()]
            if config.embedding and (not config.pretrain_fisher):
                self.embedding = nn.Parameter(torch.randn(
                    (config.category_num, config.embedding_dim)))
                optimizer_net_list.append(self.embedding)
            self.optimizer_net = optim.Adam(optimizer_net_list, config.lr)
            scheduler_list.append(optim.lr_scheduler.MultiStepLR(
                self.optimizer_net, milestones=lr_decay, gamma=config.gamma))
            self.net = DataParallel(self.net)

        if config.is_train:
            print(f'write_log_dir: {self.config.log_dir}]')
            self.writer = SummaryWriter(log_dir=self.config.log_dir)
        if not config.use_lr_decay:
            scheduler_list = None
        self.clock = utils.TrainClock(scheduler_list)

    def forward(self, data):
        gt = data.get("rot_mat").cuda()  # (b, 3, 3)
        feature, A = self.compute_feature(data, self.config.condition)

        flow = self.flow.cuda()
        rotation, ldjs = flow(gt, feature)
        losses_nll = -ldjs

        if self.config.pretrain_fisher:
            A = A.clone()
            pre_distribution = MatrixFisherN(A)  # (N, 3, 3)
            pre_ll = pre_distribution._log_prob(rotation)  # (N, )
            losses_pre_nll = -pre_ll  # probability = pre_ll + ldjs
        else:
            losses_pre_nll = 0 * losses_nll

        loss = losses_nll.mean() + losses_pre_nll.mean()

        result_dict = dict(
            loss=loss,
            losses_nll=losses_nll,
            losses_pre_nll=losses_pre_nll,
            feature=feature,
        )
        return result_dict

    def train_func(self, data):

        net_optimizers = self.train()

        result_dict = self.forward(data)

        loss = result_dict["loss"]

        for net_optimizer in net_optimizers:
            net_optimizer[1].zero_grad()

        # with torch.autograd.set_detect_anomaly(True):
        loss.backward()

        for net_optimizer in net_optimizers:
            net_optimizer[1].step()

        return result_dict

    def val_func(self, data):
        self.eval()

        if isinstance(self.flow, DataParallel):
            flow = self.flow.module
        else:
            flow = self.flow

        with torch.no_grad():
            feature, A = self.compute_feature(data, self.config.condition)
            if self.config.eval_train == 'nll':
                loss_dict = self.eval_nll(flow, data, feature, A)
            elif self.config.eval_train == 'acc':
                loss_dict = self.eval_acc(flow, data, feature, A)

            return loss_dict

    def save_ckpt(self, name=None):
        """save checkpoint during training for future restore"""
        if name is None:
            save_path = os.path.join(
                self.config.model_dir,
                "ckpt_iteration{}.pth".format(self.clock.iteration),
            )
            print(
                "[{}/{}] Saving checkpoint iteration {}...".format(
                    self.config.exp_name, self.config.date, self.clock.iteration
                )
            )
        else:
            save_path = os.path.join(
                self.config.model_dir, "{}.pth".format(name))
            print(
                "[{}/{}] Saving checkpoint {}...".format(
                    self.config.exp_name, self.config.date, name
                )
            )

        # self.flow
        flow_state_dict = self.flow.module.cpu().state_dict()

        # self.net
        if self.config.condition:
            net_state_dict = self.net.module.cpu().state_dict()

        save_dict = {
            "clock": self.clock.make_checkpoint(),
            "flow_state_dict": flow_state_dict,
            "optimizer_flow_state_dict": self.optimizer_flow.state_dict(),
        }

        if self.config.condition:
            save_dict["net_state_dict"] = net_state_dict
            save_dict["optimizer_net_state_dict"] = self.optimizer_net.state_dict()
            if self.config.category_num != 1 and self.config.embedding and (not self.config.pretrain_fisher):
                save_dict['embedding'] = self.embedding
            self.net.cuda()

        torch.save(save_dict, save_path)
        self.flow.cuda()

    def load_ckpt(self, name=None):
        """load checkpoint from saved checkpoint"""
        if os.path.isabs(name):
            load_path = name
        else:
            try:
                load_path = os.path.join(
                self.config.model_dir, "{}.pth".format(name))
            except:
                load_path = name
        
        print(load_path)
        if not os.path.exists(load_path):
            raise ValueError("Checkpoint {} not exists.".format(load_path))

        print("load_path {}".format(load_path))
        checkpoint = torch.load(load_path, map_location=torch.device("cpu"))
        print("Loading checkpoint from {} ...".format(load_path))

        if self.config.condition:
            self.net.module.load_state_dict(checkpoint["net_state_dict"])
            optimizer_net_list = [*self.net.parameters()]

            if self.config.category_num != 1 and self.config.embedding and (not self.config.pretrain_fisher):
                self.embedding = nn.Parameter(checkpoint['embedding'])
                optimizer_net_list.append(self.embedding)
                self.embedding = self.embedding.cuda()

            self.optimizer_net = optim.Adam(optimizer_net_list, self.config.lr)
            self.optimizer_net.load_state_dict(
                checkpoint["optimizer_net_state_dict"],
            )
            self.net.cuda()

        #print(checkpoint["flow_state_dict"].keys())
        self.flow.module.load_state_dict(checkpoint["flow_state_dict"])

        self.flow.cuda()
        self.optimizer_flow = optim.Adam(
            self.flow.parameters(), self.config.lr)
        self.optimizer_flow.load_state_dict(
            checkpoint["optimizer_flow_state_dict"],
        )
        self.clock.restore_checkpoint(checkpoint["clock"])

    def eval_nll(self, flow, data, feature, A):
        # compute gt_nll
        if self.config.dataset == "symsol":
            gt = data.get('rot_mat_all').cuda()
            gt_amount = gt.size(1)
            feature_2 = (
                feature[:, None, :]
                .repeat(1, gt_amount, 1)
                .reshape(-1, self.config.feature_dim + (self.config.category_num != 1)*self.config.embedding*self.config.embedding_dim)
            )
            gt_2 = gt.reshape(-1, 3, 3)
            assert gt_2.size(0) == feature_2.size(0)
            _, gt_ldjs = flow(gt_2, feature_2)
            gt_ldjs = gt_ldjs.reshape(-1, gt_amount)
            losses_ll = gt_ldjs.mean(dim=-1)
        else:
            gt = data.get('rot_mat').cuda()
            rotation, gt_ldjs = flow(gt, feature)
            losses_ll = gt_ldjs

        if self.config.pretrain_fisher:
            pre_distribution = MatrixFisherN(A.clone())  # (N, 3, 3)
            pre_ll = pre_distribution._log_prob(rotation)  # (N, )
        else:
            pre_ll = 0 * losses_ll

        loss_dict = dict(
            loss=losses_ll.mean() + pre_ll.mean(),
            losses=pre_ll + losses_ll
        )

        if self.config.dataset == "symsol":
            acc_dict = self.eval_acc(flow, data, feature, A)
            loss_dict['est_rotation'] = acc_dict['est_rotation']
            loss_dict['err_deg'] = acc_dict['err_deg']

        return loss_dict

    def eval_acc(self, flow, data, feature, A):
        batch = feature.size(0)
        feature = (
            feature[:, None, :].repeat(1, self.config.number_queries, 1).
            reshape(-1, self.config.feature_dim +
                    self.config.embedding*self.config.embedding_dim)
        )

        if self.config.pretrain_fisher:
            pre_distribution = MatrixFisherN(A.clone())  # (N, 3, 3)
            sample = pre_distribution._sample(
                self.config.number_queries).reshape(-1, 3, 3)  # sample from pre distribution
            base_ll = pre_distribution._log_prob(
                sample).reshape(batch, -1)
        else:
            sample = sd.generate_queries(self.config.number_queries).cuda()
            sample = (
                sample[None, ...].repeat(batch, 1, 1, 1).reshape(-1, 3, 3)
            )
            base_ll = torch.zeros(
                (batch, self.config.number_queries), device=feature.device)
        assert feature.size(0) == sample.size(0)

        samples, log_prob = flow.inverse(sample, feature)
        samples = samples.reshape(batch, -1, 3, 3)
        log_prob = - log_prob.reshape(batch, -1) + base_ll
        max_inds = torch.argmax(log_prob, axis=-1)
        batch_inds = torch.arange(batch)
        est_rotation = samples[batch_inds, max_inds]
        if self.config.dataset == 'symsol':
            gt = data.get('rot_mat_all').cuda()
            err_rad = utils.min_geodesic_distance_rotmats(
                est_rotation, gt)
        else:
            gt = data.get('rot_mat').cuda()
            err_rad = utils.min_geodesic_distance_rotmats(
                gt, est_rotation.reshape(batch, -1, 3, 3))
        err_deg = torch.rad2deg(err_rad)
        result_dict = dict(
            est_rotation=est_rotation,
            err_deg=err_deg,
        )
        if self.config.dataset == 'pascal3d':
            easy = torch.nonzero(data.get('easy').cuda()).reshape(-1)
            result_dict = {k: v[easy] for k, v in result_dict.items()}
        return result_dict

    def train(self):
        net_optimizers = []
        if self.config.condition:
            self.net.train()
            net_optimizers.append((self.net, self.optimizer_net))
        self.flow.train()
        net_optimizers.append((self.flow, self.optimizer_flow))
        return net_optimizers

    def eval(self):
        if self.config.condition:
            self.net.eval()
        self.flow.eval()

    def compute_feature(self, data, condition):
        if condition == 0:
            A = None
            feature = None
        else:
            img = data.get("img").cuda()
            net = self.net.cuda()
            if self.config.pretrain_fisher:
                feature, A = net(img, data.get('cate').cuda(
                ) + (1 if self.config.dataset == 'pascal3d' else 0))
                A = A.reshape(-1, 3, 3)
            else:
                feature = self.net(img)  # (b, feature_dim)
                if self.config.category_num != 1 and self.config.embedding:
                    self.embedding = self.embedding.cuda()
                    feature = torch.concat(
                        [feature, self.embedding[data.get('cate')]], dim=-1)
                A = None
        return feature, A
