from agent import get_agent
import numpy as np
import torch
from config import get_config
from dataset import get_dataloader
import utils.visualization as visualization
import utils.sd as sd
from tqdm import tqdm
from torch.nn import DataParallel
import os


def test():
    config = get_config('test')

    if config.eval == 'plot_data':
        visualize_data(config)
        return 0

    agent = get_agent(config)
    agent.load_ckpt(config.test_ckpt)
    #target_fn = get_target(config)
    if config.eval == 'nll': # compute negative log likelihood
        evaluate(config, agent)
    elif config.eval == 'sample': # visualize via sampling
        visualize_sample(config, agent)
    elif config.eval == 'pdf': # visualize via querying a grid and computing the pdf on the grid
        visualize_pdf(config, agent)


def evaluate(config, agent):
    test_loaders = get_dataloader(config.dataset, 'test', config)
    testbar = tqdm(test_loaders)
    agent.flow.eval()
    #print('evaluate')
    test_loss = []
    for i, data in enumerate(testbar):
        result_dict = agent.val_func(data)
        loss = result_dict['loss']
        # print(loss)
        test_loss.append(result_dict['losses'].cpu())

    test_loss = np.concatenate(test_loss)
    test_loss = np.mean(test_loss)
    print(test_loss)

def visualize_data(config): # visualize datasets
    root = os.path.join(config.data_dir, 'raw')
    data = np.load(root + f'/{config.category}_' + 'test' + '.npy')
    data = torch.from_numpy(data)
    fig = visualization.visualize_so3_probabilities(data,
                                                    torch.ones(data.size(0)),
                                                    ax=None,
                                                    fig=None,
                                                    #display_threshold_probability=0,
                                                    scatter_size=config.scatter_size,
                                                    # canonical_rotation=can_rotation,
                                                )
    os.makedirs('plot/raw', exist_ok=True)
    save_path = f'plot/raw/{config.category}_{config.eval}'
    print("save plot to ", save_path)
    #fig.suptitle(f'{config.target_fn}_{config.eval}_{config.ckpt}')
    fig.show()
    fig.savefig(save_path)

def visualize_sample(config, agent):
    query_roations = sd.generate_queries(config.number_queries, mode='random')
    query_roations = query_roations.cuda()

    if isinstance(agent.flow, DataParallel):
        flow = agent.flow.module.cuda()
    else:
        flow = agent.flow.cuda()
    # flow = agent.flow.cuda()
    rotations, ldjs = flow.inverse(query_roations)
    #log_prob_real = target_fn.log_prob(rotations)

    roations = rotations.cpu()
    log_pro = -ldjs.cpu()
    #print(f'kl 2 {(-ldjs-log_prob_real).mean()}')
    pro = torch.exp(ldjs)
    norm = pro.mean()
    print(f'norm:{norm}')

    fig = visualization.visualize_so3_probabilities_sample(roations,
                                                           log_pro,
                                                           ax=None,
                                                           fig=None,
                                                           display_threshold_probability=0,
                                                           scatter_size=config.scatter_size,
                                                           # canonical_rotation=torch,
                                                           )
    os.makedirs('plot/raw', exist_ok=True)
    save_path = f'plot/raw/{config.category}_{config.eval}.png'
    print("save plot to ", save_path)
    #fig.suptitle(f'{config.target_fn}_{config.eval}_{config.ckpt}')
    fig.show()
    fig.savefig(save_path)


def visualize_pdf(config, agent): # it requires large number_queries points to obtain desired pdf plot.
    query_roations = sd.generate_queries(config.number_queries, mode='random')
    query_roations = query_roations.cuda()

    if isinstance(agent.flow, DataParallel):
        flow = agent.flow.module.cuda()
    else:
        flow = agent.flow.cuda()
    #visualization.visualize_volume_target_with_mesh(flow, feature, 'plt_ax_angle', alpha = 1, threshold = 0.01, save = 1, path = 'mesh_cube')
    _, ldjs = flow(query_roations)

    query_roations = query_roations.cpu()
    log_pro = ldjs.cpu()
    pro = torch.exp(ldjs)
    norm = pro.mean()
    print(f'norm: {norm}')

    
    fig = visualization.visualize_so3_probabilities(query_roations,
                                                    log_pro,
                                                    ax=None,
                                                    fig=None,
                                                    #display_threshold_probability=0,
                                                    scatter_size=config.scatter_size,
                                                    # canonical_rotation=can_rotation,
                                                    )
    os.makedirs('plot/raw', exist_ok=True)
    save_path = f'plot/raw/{config.category}_{config.eval}.png'
    print("save plot to ", save_path)
    #fig.suptitle(f'{config.target_fn}_{config.eval}_{config.ckpt}')
    fig.show()
    fig.savefig(save_path)


if __name__ == '__main__':
    with torch.no_grad():
        test()
