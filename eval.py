import numpy as np
import torch
from config import get_config
from agent import get_agent
from dataset import get_dataloader
from torch.nn import DataParallel
from pytorch3d import transforms as trans
from tqdm import tqdm
from tqdm import trange
import utils.visualization as visualization
import utils.sd as sd
import utils.utils as utils
from utils.fisher import MatrixFisherN
import time
import random
import os


def test():
    config = get_config('test')

    agent = get_agent(config)
    agent.load_ckpt(config.test_ckpt)

    if config.category_num == 1:
        test_loaders = [get_dataloader(config.dataset, 'test', config)]
        categories = [config.category]
        test_iters = [iter(test_loaders[0])]
    else:
        _, test_loaders, categories = get_dataloader(
            config.dataset, "test", config)
        test_iters = [iter(cat_test_loader)
                      for cat_test_loader in test_loaders]
        print(categories)

    if config.eval == 'log': # default evaluation metircs
        evaluate(config, test_loaders, test_iters, categories, agent)
    ## these are ablation between different evaluation methods
    elif config.eval == 'log_inv': # evaluate by inverse sampling and find the max pdf as estimated pose
        inverse(config, test_loaders, test_iters, categories, agent)
    elif config.eval == 'log_pdf': # evaluate by querying pdf of samples and find the max pdf as estimated pose
        gradient(config, test_loaders, test_iters, categories, agent)
    elif config.eval == 'nll_grad': # evaluate by pdf+gradient descent to find the max pdf as estimated pose
        gradient(config, test_loaders, test_iters, categories, agent, gradient = True)
    else:
        ## visualize
        cate_id = categories.index(config.eval_category)
        for i, test_data in enumerate(test_iters[cate_id]):
            vis_batch = config.vis_idx // config.batch_size
            vis_number = config.vis_idx % config.batch_size
            #print(i, vis_batch, vis_number)
            if i == vis_batch:
                break
        if config.eval == 'sample':
            visualize_sample(config, test_data, agent, vis_number)
        elif config.eval == 'pdf':
            visualize_pdf(config, test_data, agent, vis_number)


def visualize_sample(config, test_data, agent, vis_number=0):
    #test_data = test_data.cuda()
    img = test_data.get('img').cuda()  # (1, 3, 224, 224)
    if config.dataset == 'symsol':
        gt = test_data.get('rot_mat_all')[vis_number, ...]  # (b, 3, 3)
    else:
        gt = test_data.get('rot_mat')[vis_number, ...]
    img_index = test_data.get('idx')[vis_number]
    #print(gt.size())

    net = agent.net.cuda()
    feature = net(img)[vis_number].reshape(1, -1)  # (1, 128)
    #print(feature.size())

    if config.category_num != 1:
        embedding = agent.embedding.cuda()
        # print(test_data.get('category'))
        # print(test_data.get('cate'))
        feature = torch.concat(
            [feature, embedding[test_data.get('cate')][vis_number].reshape(1, -1)], dim=-1)
    #print(feature.size())

    query_rotations = sd.generate_queries(config.number_queries, mode='random')
    query_rotations = query_rotations.cuda()
    feature = feature.repeat(query_rotations.size(0), 1)  # (n, 128)

    assert query_rotations.size(0) == feature.size(0)
    
    if isinstance(agent.flow, DataParallel):
            flow = agent.flow.module
    else:
        flow = agent.flow
    # flow = agent.flow.cuda()
    #visualization.visualize_volume_target_with_mesh(flow, feature, 'plt_ax_angle', alpha = 1, threshold = 0.01, save = 1, path = 'mesh_cube')
    rotations, ldjs = flow.inverse(query_rotations, feature)

    rotations = rotations.cpu()
    print(rotations.size())
    log_pro = -ldjs.cpu()
    pro = torch.exp(log_pro)

    norm = pro.mean()
    print(f'norm: {norm}')

    if config.dataset == 'symsol':
        can_rotation = gt[0]
    else:
        can_rotation = gt

    fig = visualization.visualize_so3_probabilities_sample(rotations,
                                                           log_pro,
                                                           rotations_gt=gt,
                                                           ax=None,
                                                           fig=None,
                                                           #display_threshold_probability=config.threshold,
                                                           canonical_rotation=can_rotation,
                                                           scatter_size=config.scatter_size,
                                                           )
    os.makedirs(f'plot/{config.dataset}', exist_ok=True)
    title = f'{config.dataset}/{config.eval_category}_{config.eval}_{img_index}'
    fig.suptitle(title)
    fig.show()
    fig.savefig('plot/' + title + '.png')
    print('svae plot to ', 'plot/' + title + '.png')
    print(img_index)


def visualize_pdf(config, test_data, agent, vis_number=0):
    #test_data = test_data.cuda()
    img = test_data.get('img').cuda()  # (1, 3, 224, 224)
    if config.dataset == 'symsol':
        gt = test_data.get('rot_mat_all')[vis_number, ...]  # (b, 3, 3)
    else:
        gt = test_data.get('rot_mat')[vis_number, ...]
    img_index = test_data.get('idx')[vis_number]

    net = agent.net.cuda()
    feature = net(img)[vis_number].reshape(1, -1)  # (1, 128)
    #print(feature.size())

    if config.category_num != 1:
        embedding = agent.embedding.cuda()
        # print(test_data.get('category'))
        # print(test_data.get('cate'))
        feature = torch.concat(
            [feature, embedding[test_data.get('cate')][vis_number].reshape(1, -1)], dim=-1)
    #print(feature.size())

    query_rotations = sd.generate_queries(config.number_queries, mode='random')
    query_rotations = query_rotations.cuda()
    feature = feature.repeat(query_rotations.size(0), 1)  # (n, 128)

    assert query_rotations.size(0) == feature.size(0)

    flow = agent.flow.cuda()
    #visualization.visualize_volume_target_with_mesh(flow, feature, 'plt_ax_angle', alpha = 1, threshold = 0.01, save = 1, path = 'mesh_cube')
    rotations, ldjs = flow(query_rotations, feature)

    query_rotations = query_rotations.cpu()
    log_pro = ldjs.cpu()
    pro = torch.exp(log_pro)

    if config.dataset == 'symsol':
        can_rotation = gt[0]
    else:
        can_rotation = gt

    norm = pro.mean()
    print(f'norm: {norm}')

    fig = visualization.visualize_so3_probabilities(query_rotations,
                                                    log_pro,
                                                    rotations_gt=gt,
                                                    ax=None,
                                                    fig=None,
                                                    #display_threshold_probability=config.threshold,
                                                    canonical_rotation=can_rotation,
                                                    scatter_size=config.scatter_size,
                                                    )
    os.makedirs(f'plot/{config.dataset}', exist_ok=True)
    title = f'{config.dataset}/{config.eval_category}_{config.eval}_{img_index}'
    fig.suptitle(title)
    fig.show()
    fig.savefig('plot/' + title + '.png')
    print('svae plot to ', 'plot/' + title + '.png')
    print(img_index)


def evaluate(config, test_loaders, test_iters, categories, agent):

    if config.dataset == 'symsol':
        agent.net.eval()
        agent.flow.eval()
        categories_log_likelihood_lst = []
        print(f'==== exp: {config.exp_name} ====')
        for cate_id in range(len(categories)):
            log_likelihood_lst = []
            testbar = tqdm(test_loaders[cate_id])
            for i, data in enumerate(testbar):
                img = data.get('img').cuda()
                cat = data.get('cate')
                #print(cat, cate_id)
                net = agent.net.cuda()
                flow = agent.flow.cuda()

                feature = net(img)

                if config.embedding:
                    embedding = agent.embedding.cuda()
                    feature = torch.concat(
                        [feature, embedding[data.get('cate')]], dim=-1)

                gt = data.get('rot_mat_all').cuda()
                gt_size = gt.size(1)
                gt = gt.reshape(-1, 3, 3)
                feature = (
                    feature[:, None, :].repeat(1, gt_size, 1).reshape(-1, config.feature_dim + (
                        config.category_num != 1)*config.embedding_dim*config.embedding)
                )
                _, ldjs = flow(gt, feature)
                ldjs = ldjs.reshape(-1, gt_size)

                # print(feature.size())

                ldjs = torch.mean(ldjs, dim=-1)

                log_likelihoods = ldjs
                log_likelihood = log_likelihoods.mean()
                log_likelihood_lst.append(log_likelihood.cpu())

            log_likelihood_lst = np.array(log_likelihood_lst)
            categories_log_likelihood_lst.append(log_likelihood_lst)
            print(f'==== category: {categories[cate_id]} ====')
            print(f'{categories[cate_id]}_ll: {np.mean(log_likelihood_lst)}')

        categories_log_likelihood_lst = np.array(categories_log_likelihood_lst)
        print(f'loss_mean: {np.mean(categories_log_likelihood_lst)}')

    print(f'==== exp: {config.exp_name} ====')
    for top_k in [1,]:  # 2, 4]: # top_k
        categories_err_mean = []
        categories_err_median = []
        categories_acc_15 = []
        categories_acc_30 = []
        count_number = 0
        count_time = 0
        for cate_id in range(len(categories)):
            # print(cate_id)
            err_lst = []

            testbar = tqdm(test_loaders[cate_id])
            for i, data in enumerate(testbar):
                #print(data.get('rot_mat').size())
                # print(data.get('cate'))
                val_dict = agent.val_func(data)

                err_deg = val_dict['err_deg']
                if config.dataset == 'pascal':
                    print('evaluate easy.')
                err_lst.append(err_deg.detach().cpu().numpy())
            err_lst = np.concatenate(err_lst, 0)
            acc_15 = utils.acc(err_lst, 15)
            acc_30 = utils.acc(err_lst, 30)

            categories_err_mean.append(np.mean(err_lst))
            categories_err_median.append(np.median(err_lst))
            categories_acc_15.append(acc_15)
            categories_acc_30.append(acc_30)
            print(
                f'==== category: {categories[cate_id]} ====')
            print(f'{categories[cate_id]}_mean: {np.mean(err_lst)}')
            print(f'{categories[cate_id]}_median: {np.median(err_lst)}')
            print(f'{categories[cate_id]}_acc15: {acc_15}')
            print(f'{categories[cate_id]}_acc30: {acc_30}')

        categories_err_mean = np.array(categories_err_mean)
        categories_err_median = np.array(categories_err_median)
        categories_acc_15 = np.array(categories_acc_15)
        categories_acc_30 = np.array(categories_acc_30)
        print(f'total err_mean {np.mean(categories_err_mean)}')
        print(f'total err_median {np.mean(categories_err_median)}')
        print(f'total acc15 {np.mean(categories_acc_15)}')
        print(f'toatl acc30 {np.mean(categories_acc_30)}')
        #print(f'inference_time: {count_time/count_number}')

def inverse(config, test_loaders, test_iters, categories, agent):
    # only use for time query
    print(f'==== exp: {config.exp_name} ====')
    agent.eval()
    if isinstance(agent.flow, DataParallel):
            flow = agent.flow.module
    else:
        flow = agent.flow
    #for top_k in [1,]:  # 2, 4]: # top_k
    categories_err_mean = []
    categories_err_median = []
    categories_acc_15 = []
    categories_acc_30 = []
    #count_number = 0
    evaluate_time = []
    evaluate_size = []
    evaluate_iter = []
    with torch.no_grad():
        for cate_id in range(len(categories)):
            # print(cate_id)
            err_lst = []
            evaluate_per_size = 0
            evaluate_per_time = 0
            evaluate_per_iter = 0
            testbar = tqdm(test_loaders[cate_id])
            #start_time = time.time()
            for i, data in enumerate(testbar):
                # print(data.get('cate'))
                start_time = time.time()
                feature, A = agent.compute_feature(data, agent.config.condition)
                # print(feature)

                batch = feature.size(0)
                feature = (
                    feature[:, None, :].repeat(1, config.number_queries, 1).
                    reshape(-1, config.feature_dim +
                            config.embedding*config.embedding_dim)
                )
                if config.pretrain_fisher:
                    pre_distribution = MatrixFisherN(A.clone())  # (N, 3, 3)
                    sample = pre_distribution._sample(
                        config.number_queries).reshape(-1, 3, 3)  # sample from pre distribution
                    base_ll = pre_distribution._log_prob(
                        sample).reshape(batch, -1)
                else:
                    sample = sd.generate_queries(config.number_queries).cuda()
                    sample = (
                        sample[None, ...].repeat(batch, 1, 1, 1).reshape(-1, 3, 3)
                    )
                    base_ll = torch.zeros(
                        (batch, config.number_queries), device=feature.device)
                #assert feature.size(0) == sample.size(0)
                samples, log_prob = flow.inverse(sample, feature)
                samples = samples.reshape(batch, -1, 3, 3)
                log_prob = - log_prob.reshape(batch, -1) + base_ll
                # warning: top_k = 1
                max_inds = torch.argmax(log_prob, axis=-1)
                batch_inds = torch.arange(batch)
                est_rotation = samples[batch_inds, max_inds]
                est_rotation = est_rotation.reshape(batch, -1, 3, 3)
                
                gt = data.get('rot_mat').cuda()

                err_rad = utils.min_geodesic_distance_rotmats(gt, est_rotation)
                err_deg = torch.rad2deg(err_rad)

                if config.dataset == 'pascal':
                    print('evaluate easy.')
                    easy = torch.nonzero(data.get('easy').cuda()).reshape(-1)
                    err_deg = err_deg[easy]
                end_time = time.time()
                evaluate_per_time += end_time - start_time
                evaluate_per_size += batch
                evaluate_per_iter += 1
                err_lst.append(err_deg.detach().cpu().numpy())
            err_lst = np.concatenate(err_lst, 0)
            acc_15 = utils.acc(err_lst, 15)
            acc_30 = utils.acc(err_lst, 30)

            categories_err_mean.append(np.mean(err_lst))
            categories_err_median.append(np.median(err_lst))
            categories_acc_15.append(acc_15)
            categories_acc_30.append(acc_30)

            evaluate_size.append(evaluate_per_size)
            evaluate_time.append(evaluate_per_time)
            evaluate_iter.append(evaluate_per_iter)
            print(
                f'==== category: {categories[cate_id]} ====')
            print(f'{categories[cate_id]}_mean: {np.mean(err_lst)}')
            print(f'{categories[cate_id]}_median: {np.median(err_lst)}')
            print(f'{categories[cate_id]}_acc15: {acc_15}')
            print(f'{categories[cate_id]}_acc30: {acc_30}')
            print(f'evaulate total size: {evaluate_per_size}')
            print(f'evaulate time per size: {evaluate_per_time / evaluate_per_size}')
            print(f'evaulate time per iter: {evaluate_per_time / evaluate_per_iter}')


        categories_err_mean = np.array(categories_err_mean)
        categories_err_median = np.array(categories_err_median)
        categories_acc_15 = np.array(categories_acc_15)
        categories_acc_30 = np.array(categories_acc_30)
        evaluate_size = np.array(evaluate_size)
        evaluate_time = np.array(evaluate_time)
        print(f'total err_mean {np.mean(categories_err_mean)}')
        print(f'total err_median {np.mean(categories_err_median)}')
        print(f'total acc15 {np.mean(categories_acc_15)}')
        print(f'toatl acc30 {np.mean(categories_acc_30)}')
        print(f'total evaluate time {np.sum(evaluate_time)}')
        print(f'average per item time: {np.sum(evaluate_time) / np.sum(evaluate_size)}')
        print(f'average per iter time: {np.sum(evaluate_time) / np.sum(evaluate_iter)}')
            #print(f'inference_time: {count_time/count_number}')


def gradient(config, test_loaders, test_iters, categories, agent, gradient=False):
    #TODO(only work for Modelnet/Pascal)
    print(f'==== exp: {config.exp_name} ====')
    #for top_k in [1,]:  # 2, 4]: # top_k
    categories_err_mean = []
    categories_err_median = []
    categories_acc_15 = []
    categories_acc_30 = []
    evaluate_time = []
    evaluate_size = []
    evaluate_iter = []
    agent.eval()
    if isinstance(agent.flow, DataParallel):
        flow = agent.flow.module
    else:
        flow = agent.flow
    with torch.no_grad():
        for cate_id in range(len(categories)):
            # print(cate_id)
            err_lst = []
            evaluate_per_size = 0
            evaluate_per_time = 0
            evaluate_per_iter = 0

            testbar = tqdm(test_loaders[cate_id])
            for i, data in enumerate(testbar):
                # print(data.get('cate'))
                start_time = time.time()
                feature, A = agent.compute_feature(data, agent.config.condition)
                # print(feature)
                if config.pretrain_fisher:
                    pre_distribution = MatrixFisherN(A.clone())  # (N, 3, 3)

                batch = feature.size(0)
                sample_rotations = sd.generate_queries(config.number_queries, mode = 'grid').cuda()
                num_grids, _, _ = sample_rotations.size()
                random_rot = trans.random_rotation().cuda()
                samples = sample_rotations @ random_rot
                est_rotation = torch.empty((batch, 3, 3), dtype = feature.dtype, device = feature.device)

                #max_inds = []
                for i in trange(batch):
                    batch_grids = 500_000
                    
                    ldjss = []    
                    for j in range((num_grids + batch_grids - 1) // batch_grids):
                        samples_batch = samples[j * batch_grids: min((j + 1) * batch_grids, num_grids)]
                        feature_batch = (feature[i])[None,...].repeat(min(batch_grids, num_grids - j*batch_grids), 1)
                        # print(samples_batch.size())
                        # print(feature_batch.size())
                        base_samples, ldjs_i = flow(samples_batch, feature_batch)
                        if config.pretrain_fisher:
                                # sample from pre distribution
                            base_ll = pre_distribution._log_prob(
                                base_samples).reshape(-1, )
                            ldjs_i += base_ll
                        ldjss.append(ldjs_i)
                    ldjss = torch.cat(ldjss, dim = 0)
                    max_inds_i = torch.argmax(ldjss)
                    est_rotation[i] = samples[max_inds_i]
                
                if gradient:
                    update_step_size = 1e-4
                    number_optimization_steps = 100
                    query_quaternions = trans.matrix_to_quaternion(est_rotation)
                    with torch.set_grad_enabled(True):
                        for step in range(number_optimization_steps):
                            query_quaternions = torch.autograd.Variable(query_quaternions, requires_grad = True)
                            optimizer_eval = torch.optim.Adam([query_quaternions], lr = update_step_size)
                            query_rotations_inp = trans.quaternion_to_matrix(query_quaternions) # (b, 3, 3)
                            _, ldjs = flow(query_rotations_inp, feature) # warning: training = False? # feature (b, feature_dim)
                            grad_loss = - ldjs.mean() # -probability 
                            optimizer_eval.zero_grad()
                            grad_loss.backward()
                            optimizer_eval.step()
                            query_quaternions = query_quaternions / torch.norm(query_quaternions, dim=-1, keepdim=True)
                            #print(f'loss {grad_loss} result {query_quaternions[:3]}')
                    est_rotation = trans.quaternion_to_matrix(query_quaternions)

                est_rotation = est_rotation.reshape(batch, -1, 3, 3)
                
                gt = data.get('rot_mat').cuda()

                err_rad = utils.min_geodesic_distance_rotmats(gt, est_rotation)
                err_deg = torch.rad2deg(err_rad)

                if config.dataset == 'pascal':
                    print('evaluate easy.')
                    easy = torch.nonzero(data.get('easy').cuda()).reshape(-1)
                    err_deg = err_deg[easy]
                
                end_time = time.time()
                evaluate_per_time += end_time - start_time
                evaluate_per_size += batch
                evaluate_per_iter += 1

                err_lst.append(err_deg.detach().cpu().numpy())
            err_lst = np.concatenate(err_lst, 0)
            acc_15 = utils.acc(err_lst, 15)
            acc_30 = utils.acc(err_lst, 30)

            categories_err_mean.append(np.mean(err_lst))
            categories_err_median.append(np.median(err_lst))
            categories_acc_15.append(acc_15)
            categories_acc_30.append(acc_30)
            evaluate_size.append(evaluate_per_size)
            evaluate_time.append(evaluate_per_time)
            evaluate_iter.append(evaluate_per_iter)
            print(
                f'==== category: {categories[cate_id]} ====')
            print(f'{categories[cate_id]}_mean: {np.mean(err_lst)}')
            print(f'{categories[cate_id]}_median: {np.median(err_lst)}')
            print(f'{categories[cate_id]}_acc15: {acc_15}')
            print(f'{categories[cate_id]}_acc30: {acc_30}')
            print(f'evaulate total size: {evaluate_per_size}')
            print(f'evaulate time per size: {evaluate_per_time / evaluate_per_size}')
            print(f'evaulate time per iter: {evaluate_per_time / evaluate_per_iter}')


        categories_err_mean = np.array(categories_err_mean)
        categories_err_median = np.array(categories_err_median)
        categories_acc_15 = np.array(categories_acc_15)
        categories_acc_30 = np.array(categories_acc_30)
        print(f'total err_mean {np.mean(categories_err_mean)}')
        print(f'total err_median {np.mean(categories_err_median)}')
        print(f'total acc15 {np.mean(categories_acc_15)}')
        print(f'toatl acc30 {np.mean(categories_acc_30)}')
        print(f'total evaluate time {np.sum(evaluate_time)}')
        print(f'average per item time: {np.sum(evaluate_time) / np.sum(evaluate_size)}')
        print(f'average per iter time: {np.sum(evaluate_time) / np.sum(evaluate_iter)}')

            #print(f'inference_time: {count_time/count_number}')



if __name__ == '__main__':
    with torch.no_grad():
        test()
