import sys
from os.path import dirname, abspath, join
from tqdm import tqdm
import numpy as np
import torch
from config import get_config
from dataset import get_dataloader
from agent import get_agent
from utils.utils import cycle, dict_get, acc


def main():
    # create experiment config containing all hyperparameters
    config = get_config("train")
    torch.manual_seed(config.RD_SEED)
    np.random.seed(config.RD_SEED)

    # create dataloader
    train_loader = get_dataloader(config.dataset, "train", config)
    test_loader = get_dataloader(config.dataset, "test", config)
    test_iter = iter(test_loader)

    # create network and training agent
    agent = get_agent(config)

    if config.cont:
        # recover training
        agent.load_ckpt(config.ckpt)
        agent.clock.tock()

        # for param_group in agent.optimizer.param_groups:
        #    param_group["lr"] = config.lr

    # start training
    clock = agent.clock

    while True:
        # begin iteration
        pbar = tqdm(train_loader)
        for b, data in enumerate(pbar):
            # train step
            result_dict = agent.train_func(data)
            loss = result_dict["loss"]

            if agent.clock.iteration % config.log_frequency == 0:
                # print(fisher_dict["count_list"])
                agent.writer.add_scalar(
                    "train/lr",
                    agent.optimizer_flow.param_groups[0]["lr"],
                    clock.iteration,
                )
                agent.writer.add_scalar(
                    "train/loss", loss, clock.iteration
                )

            pbar.set_description("EPOCH[{}][{}]".format(
                clock.epoch, clock.minibatch))
            pbar.set_postfix({"loss": loss.item()})

            clock.tick()

            # evaluation
            if clock.iteration % config.val_frequency == 0:
                test_loss = []

                for i in range(10):
                    try:
                        data = next(test_iter)
                    except:
                        test_iter = iter(test_loader)
                        data = next(test_iter)
                    result_dict = agent.val_func(data)
                    loss = result_dict["loss"]
                    test_loss.append(result_dict["losses"].cpu())

                test_loss = np.concatenate(test_loss, 0)
                test_loss = np.mean(test_loss)
                print(test_loss)
                # fisher_test_entropy = np.concatenate(fisher_test_entropy, 0)
                agent.writer.add_scalar(
                    "test/loss", test_loss, clock.iteration
                )

            # save checkpoint
            if clock.iteration % config.save_frequency == 0:
                agent.save_ckpt()

        clock.tock()

        if clock.iteration > config.max_iteration:
            break


if __name__ == "__main__":
    main()
