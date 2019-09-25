import os
import torch
from settings import Settings
from quicknat import QuickNat
from utils.data_utils import get_imdb_dataset
from solver import Solver
import tables


def load_data(data_params):
    print("Loading dataset")
    train_data, test_data = get_imdb_dataset(data_params)
    print("Train size: %i" % len(train_data))
    print("Test size: %i" % len(test_data))
    return train_data, test_data


def train(train_params, common_params, data_params, net_params):

    train_data, test_data = load_data(data_params)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_params['train_batch_size'],
                                               shuffle=False, num_workers=4, pin_memory=False)

    quicknat_model = QuickNat(net_params)
    #print(quicknat_model)

    solver = Solver(quicknat_model,
                    device=common_params['device'],
                    num_class=net_params['num_class'],
                    optim_args={"lr": train_params['learning_rate'],
                                "betas": train_params['optim_betas'],
                                "eps": train_params['optim_eps'],
                                "weight_decay": train_params['optim_weight_decay']
                                },
                    model_name=common_params['model_name'],
                    labels=data_params['labels'],
                    num_epochs=train_params['num_epochs'],
                    lr_scheduler_step_size=train_params['lr_scheduler_step_size'],
                    lr_scheduler_gamma=train_params['lr_scheduler_gamma'])

    solver.train(train_loader)
    # final_model_path = os.path.join(common_params['save_mode_dir'], train_params['final_model_file'])
    # quicknat_model.save_model(final_model_path)
    # print("Final model saved @ " + str(final_model_path))


if __name__ == '__main__':
    settings = Settings()

    common_params = settings['COMMON']
    data_params = settings['DATA']
    net_params = settings['NETWORK']
    train_params = settings['TRAINING']

    train(train_params, common_params, data_params, net_params)
