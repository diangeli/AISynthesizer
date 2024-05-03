# add dependencies
import os
import torch
import numpy as np

def save(model_name, best_model, train_loss, test_loss, accuracy):
    path = os.path.join('results', model_name)
    if not os.path.exists(path):
        os.makedirs(path)

    np.save(os.path.join(path, 'train_loss.npy'), np.array(train_loss))
    np.save(os.path.join(path, 'test_loss.npy'), np.array(test_loss))
    np.save(os.path.join(path, 'accuracy.npy'), np.array(accuracy))
    torch.save(best_model.state_dict(), os.path.join(path, 'model.pt'))
    return

