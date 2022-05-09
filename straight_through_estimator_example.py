from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import copy
import random
import math
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

def train_valid_loaders(dataset, batch_size, train_split=0.8, shuffle=True, seed=42):
    """
    Divise un jeu de données en ensemble d'entraînement et de validation et retourne pour chacun un DataLoader PyTorch.
    Args:
        dataset (torch.utils.data.Dataset): Un jeu de données PyTorch
        batch_size (int): La taille de batch désirée pour le DataLoader
        train_split (float): Un nombre entre 0 et 1 correspondant à la proportion d'exemple de l'ensemble
            d'entraînement.
        shuffle (bool): Si les exemples sont mélangés aléatoirement avant de diviser le jeu de données.
        seed (int): Le seed aléatoire pour que l'ordre des exemples mélangés soit toujours le même.
    Returns:
        Tuple (DataLoader d'entraînement, DataLoader de test).
    """
    num_data = len(dataset)
    indices = np.arange(num_data)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    split = math.floor(train_split * num_data)
    train_idx, valid_idx = indices[:split], indices[split:]

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader


def load_mnist(low, high, random_state = 42):
    rand = random_state
    random_state = check_random_state(random_state)
    X_low = np.loadtxt(join("data/mnist", f"mnist_{low}")) / 255
    y_low = -1 * np.ones(X_low.shape[0])

    X_high = np.loadtxt(join("data/mnist", f"mnist_{high}")) / 255
    y_high = np.ones(X_high.shape[0])
    
    X = np.vstack((X_low, X_high))
    y = np.hstack((y_low, y_high)).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = random_state)
    print(f"\nDataset MNIST{low}{high}, random seed {rand}; TEST Pos:{len(y_test[np.where(y_test == 1)])}, Neg:{len(y_test[np.where(y_test == -1)])}.\n")
    
    train = []
    test = []
    for i in range(len(X_train)) :
        if y_train[i] == 1 :
            train.append((X_train[i],[-1,1]))
        else :
            train.append((X_train[i],[1,-1]))
    for i in range(len(X_test)) :
        if y_test[i] == 1 :
            test.append((X_test[i],[-1,1]))
        else :
            test.append((X_test[i],[1,-1]))
        
    return train, test


def linear_loss(pred_y, y):
    """Linear loss function."""
    return torch.mean((1 -(y * pred_y)) / 2)


class Straight_through_bnn(nn.Module):
    """
    Implementation of the Straigh trough estimator; basically a sign function 
    for which the gradient passes trough the sign function (if the input value
    isn't too big) during the backward phase of the training : 
    
    g_a = 1_{|a| < 1} * g_a^b
    
    """
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input + torch.sign(input+1e-8) - input.detach()
        out[torch.abs(input) > 1] = out[torch.abs(input) > 1].detach()
        return out

    
class MnistNet(nn.Module):
    def __init__(self, num_neur_ent, num_neur, num_class, num_hid_lay, algo, beta, T, nbatch, softmax = False, BN = True, a = 2 ** 0):
        super(MnistNet, self).__init__()
        self.layers = torch.nn.ModuleList()
        if num_hid_lay == 0 :
            self.layers.append(nn.Linear(num_neur_ent, num_class))
        else :
            self.layers.append(nn.Linear(num_neur_ent, num_neur))
            if BN == True :
                self.layers.append(nn.BatchNorm1d(num_neur))
            if algo == 'bnn' :
                self.layers.append(Straight_through_bnn())
            elif algo == 'bc' :
                self.layers.append(nn.ReLU())
        for i in range(num_hid_lay-1) :
            self.layers.append(nn.Linear(num_neur, num_neur))
            if BN == True :
                self.layers.append(nn.BatchNorm1d(num_neur))
            if algo == 'bnn' :
                self.layers.append(Straight_through_bnn())
            elif algo == 'bc' :
                self.layers.append(nn.ReLU())
        if num_hid_lay > 0 :
            self.layers.append(nn.Linear(num_neur, num_class))
        
        if softmax == True :
            self.layers.append(nn.Softmax(dim=1))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def init_weights(init, model, algo, data = False) :
    i = 0
    for layer in model.layers :
        i += 1
        if isinstance(layer, nn.Linear):
            if init == 'kaiming_unif' :
                nn.init.kaiming_uniform_(layer.weight.data, nonlinearity = 'relu')
            elif init == 'kaiming_norm' :
                nn.init.kaiming_normal_(layer.weight.data, nonlinearity = 'relu')
            elif init == 'xavier_unif' :
                nn.init.xavier_uniform_(layer.weight.data)
            elif init == 'xavier_norm' :
                nn.init.xavier_normal_(layer.weight.data)
            elif init == 'unif' :
                if i == 1 :
                    dat = np.linspace(min(data)[0],max(data)[0], num = len(layer.bias.data)+2)
                    for j in range(len(layer.bias.data)) :
                        layer.bias.data[j] = dat[j+1]
                    nn.init.ones_(layer.weight.data)
                    layer.weight.data *= -1
                else :
                    nn.init.uniform_(layer.weight.data, a = 0, b = 1)
                    nn.init.uniform_(layer.weight.data, a = 0, b = 1)
                    
            elif init == 'rand_unif' :
                if i == 1 :
                    nn.init.uniform_(layer.weight.data, a = -1, b = 1)
                    on = torch.ones(layer.weight.data.size())/2
                    on = torch.bernoulli(on)
                    ver_1 = on == 1
                    ver_0 = on == 0
                    layer.weight.data = layer.weight.data * ver_1 + layer.weight.data ** (-1) * ver_0
                    for j in range(len(layer.bias.data)) :
                        layer.bias.data[j] = (torch.round(torch.rand(1))*2-1)*max(data)[0][0]*0.25#*torch.min(torch.abs(layer.weight.data[j]))*0.5#*torch.rand(1)#*layer.weight.data[j][0]
                else :
                    nn.init.uniform_(layer.bias.data, a = -10, b = 10)
                    nn.init.uniform_(layer.weight.data, a = -100, b = 100)

def train(model, job, dataset, algo, bina, optim_algo, momentum, n_epoch, batch_size, num_class, 
          criterion, DEVICE, start_lr, end_lr, factor, dec_lr, patience, early_stop, 
          early_stop_value, nesterov, reg_type=None, lambd=0, use_gpu=False, dat=False):
    """
    Entraîne un réseau de neurones de classification pour un certain nombre d'epochs 
    avec PyTorch.
    
    Args:
        model (nn.Module): Un réseau de neurones instancié avec PyTorch.
        dataset (Dataset): Un jeu de données PyTorch.
        n_epoch (int): Le nombre d'epochs.
        batch_size (int): La taille des batchs.
        learning_rate (float): Le taux d'apprentissage pour SGD.
        use_gpu (bool): Si les données doivent être envoyées sur GPU.
    
    Returns:
        Retourne un objet History permettant de faire des graphiques
        de l'évolution de l'entraînement.
    """
    # La classe History vient de deeplib. Elle va nous permettre de faire les graphiques
    # donnant l'évolution de la perte et de l'exactitude (accuracy).
    #history = History()
    
    # La fonction de perte que nous utilisons ici est l'entropy croisée
    # L'optimiseur que nous utilisons ici est le classique SGD.
    # Des liens vers la documentation de PyTorch sont en commentaires.
    if optim_algo == 'sgd' :
        optimizer = torch.optim.SGD(model.parameters(), lr=start_lr, 
                                    momentum = momentum, nesterov = nesterov)
    elif optim_algo == 'adagrad' :
        optimizer = torch.optim.Adagrad(model.parameters(), lr=start_lr)
    elif optim_algo == 'adam' :
        optimizer = torch.optim.Adam(model.parameters(), lr=start_lr)
        
    if dec_lr == 'plateau' :
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer = optimizer, factor = factor, patience=patience, threshold=0.0001, verbose = True)
    # La fonction train_valid_loaders vient de deeplib. Elle nous retourne deux DataLoaders:
    # un pour l'ensemble d'entraînement et un pour l'ensemble de test. Essentiellement,
    # DataLoader est une classe de PyTorch nous permettant de faire des batchs avec la taille
    # désirée. La fonction train_valid_loaders effectue la répartition aléatoire des exemples
    # en entraînement et en validation.
    #print(dataset)
    
    train_loader, valid_loader = train_valid_loaders(dataset, batch_size=batch_size)
    train_loader_copy, valid_loader_copy = train_valid_loaders(dataset, batch_size=batch_size)
            
            
    # On crée la variable à même de pouvoir arrêter l'entrainement dans le cas où il 
    #   n'y à plus d'amélioration dans l'accuracy en entrainement depuis un moment
    #   (early stopping).
    best_val_acc = 0
    best_train_acc = 0
    best_val_loss = 1e10
    best_train_loss = 1e10
    early_stop_count = 0
    # C'est ici que la boucle d'entraînement commence. On va donc faire n_epochs epochs.
    for i in range(n_epoch):
        #time_1 = time()
        #if i % 10 == 0 :
        #    show(model, algo, bina, dat, batch_size, num_class, criterion, DEVICE, use_gpu=False, already_load = False, i = False)
        # Les réseaux de neurones avec PyTorch ont un méthode train() en plus d'une méthode 
        # eval(). Ces deux méthodes indiquent au réseau s'il est en entraînement ou bien en test.
        # Ceci permet au réseau de modifier son comportement en fonction. On va le voir plus tard
        # certaines couches agissent différemment selon le mode, nommément le dropout et la 
        # batch normalization.
        model.train()
        
        # La prochaine ligne active simplement le calcul du gradient. Le gradient est ce qui va
        # nous permettre de mettre à jour les poids du réseau de neurones. En test, le calcul du
        # gradient sera désactivé étant qu'il n'est pas nécessaire et qu'il peut engendrer des 
        # fuites de mémoire si la rétro-propagation n'est pas effectuée.
        with torch.enable_grad():
            #print(len(train_loader))
            # À chaque epoch, on parcourt l'ensemble d'entraînement au complet via le DataLoader
            # qui nous le retourne en batch (x, y) comme mentionné plus haut. La variable inputs 
            # correspond donc à une batch d'exemples (x) et targets correspond à une batch 
            # d'étiquettes (y).
            #print(load)
            for inputs, targets in train_loader :
                #show(model, algo, bina, dat, batch_size, num_class, criterion, DEVICE, use_gpu=False, already_load = False)
                #input()
                # On envoie les exemples et leurs étiquettes sur GPU via la méthode cuda() si 
                # demandé.
                #if num_class == 1 :
                #    targets = (targets[0].long()+1)/2
                if num_class in [1,2] :
                    if not torch.is_tensor(targets) :
                        tar_1 = torch.reshape(targets[0],(len(targets[0]),1)) 
                        tar_2 = torch.reshape(targets[1],(len(targets[1]),1))
                        targets = torch.hstack((tar_1, tar_2))
                    else :
                        targets = torch.reshape(targets,(len(targets),1)) 
                    if not torch.is_tensor(inputs) :
                        inp = inputs[0]
                        for k in range(len(inputs)-1) :
                            inp = torch.vstack((inp,inputs[k+1]))
                        inp = torch.transpose(inp, 0, 1)
                        inputs = torch.clone(inp)
                    
                inputs = inputs.float()
                
                # La méthode zero_grad() de l'optimiseur permet de mettre la valeur du gradient
                # à zéro de façon à effacer le gradient calculé auparavant. Si ceci n'était pas 
                # fait, le nouveau gradient serait additionné à l'ancien gradient ce qui poserait
                # problème.
                optimizer.zero_grad()
                
                if bina in ['sto', 'det'] :
                    param_sauv = []
                    for layer in model.layers :
                        if isinstance(layer, nn.Linear):
                            param_sauv.append(torch.clone(layer.weight.data))
                            if bina == 'det' :
                                layer.weight.data = torch.sign(layer.weight.data)
                            elif bina == 'sto' :
                                p = torch.clip((layer.weight.data+1)/2,0,1)
                                layer.weight.data = torch.sign(torch.bernoulli(p)-0.5)                   

                # C'est ici que finalement le réseau de neurones est appelé. On lui donne en entrée
                # un exemple et en sortie il nous donne ses prédictions (ici, des scores de 
                # classification).
                #print(inputs)
                if str(DEVICE) == 'cuda' :
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    model = model.cuda()  
                
                output = model(inputs)
                #print(output)
                # Une fois nos prédictions obtenues, on calcule la perte avec la fonction de perte 
                # qui nous retourne un tenseur scalaire.
                targets = targets.long()
                if str(DEVICE) == 'cuda' :
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                    model = model.cuda() 
                loss = criterion(output, targets)
                
                # Ce tenseur scalaire nous permet de calculer le gradient. C'est ce que la méthode
                # backward() vient faire pour nous via la rétropropagation.
                loss.backward()
                
                if bina in ['sto', 'det'] :
                    j = 0
                    for layer in model.layers :
                        if isinstance(layer, nn.Linear):
                            layer.weight.data = torch.clone(param_sauv[j])
                            j += 1
                            
                optimizer.step()
        # Après chaque epoch d'entraînement, on va venir calculer la perte et l'exactitude 
        # (accuracy) sur l'ensemble d'entraînement et de validation.
        
        if str(DEVICE) == 'cuda' :
            model = model.cuda()   
        train_acc, train_loss = validate(model, job, algo, bina, train_loader_copy, num_class, criterion, DEVICE, use_gpu)
        val_acc, val_loss = validate(model, job, algo, bina, valid_loader, num_class, criterion, DEVICE, use_gpu)
        
        if job == 'classification' :
            print(f'Epoch {i+1} - Train acc: {train_acc:.2f} - Val acc: {val_acc:.2f}')
        elif job in ['regression', 'pruning'] :
            print(f'Epoch {i+1} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}')
        
        if dec_lr == 'exponential' :
            scheduler.step()
        if dec_lr == 'plateau' :
            scheduler.step(val_loss)
        if i == 1 : 
            cop = copy.deepcopy(model)
        if job == 'classification' :
            if val_acc > best_val_acc + early_stop_value :
                best_val_acc = val_acc
                best_train_acc = train_acc
                early_stop_count = -1
                cop = copy.deepcopy(model)
            early_stop_count += 1
            if early_stop != False :
                if early_stop_count >= early_stop :
                    break
            if train_acc > 99.99 :
                break
        elif job in ['regression', 'pruning'] :
            if val_loss < best_val_loss - early_stop_value :
                best_val_loss = val_loss                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
                best_train_loss = train_loss
                early_stop_count = -1
                cop = copy.deepcopy(model)
            early_stop_count += 1
            if early_stop != False :
                if early_stop_count >= early_stop :
                    break

    for name, param in model.named_parameters() :
        param.requires_grad = True
    
    if job == 'classification' :
        return cop, best_train_acc, best_val_acc, ''
    elif job  == 'regression' :
        return cop, best_train_loss, best_val_loss, ''


def validate(model, job, algo, bina, valid_loader, num_class, criterion, DEVICE, use_gpu=False, return_data = False):
    """
    Test un réseau de neurones de classification pour un certain nombre d'epochs 
    avec PyTorch.
    
    Args:
        model (nn.Module): Un réseau de neurones instancié avec PyTorch.
        valid_loader (DataLoader): Un DataLoader PyTorch tel qu'instancié dans train() 
            et test().
        use_gpu (bool): Si les données doivent être envoyées sur GPU.
            
    Returns:
        Retourne un tuple (exactitude, perte) pour les données du DataLoader en argument.
    """
    
    # Les étapes de la fonction validate est très similaire à celle de la fonction train.
    # Essentiellement, le réseau est mis en mode évaluation au lieu d'entraînement et le 
    # calcul du gradient est désactivé. Il n'y a bien sûr pas d'utilisation d'un optimiseur.
    true = []
    pred = []
    inp = []
    val_loss = []
    
    if num_class == 2 :
        criterion = linear_loss
    else :
        criterion = nn.CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        i = -1
        for inputs, targets in valid_loader :
            i += 1
            if num_class in [1,2] :
                if not torch.is_tensor(targets) :
                    tar_1 = torch.reshape(targets[0],(len(targets[0]),1)) 
                    tar_2 = torch.reshape(targets[1],(len(targets[1]),1))
                    targets = torch.hstack((tar_1, tar_2))
                else :
                    targets = torch.reshape(targets,(len(targets),1)) 
                if not torch.is_tensor(inputs) :
                    inpp = inputs[0]
                    for k in range(len(inputs)-1) :
                        inpp = torch.vstack((inpp,inputs[k+1]))
                    inpp = torch.transpose(inpp, 0, 1)
                    inputs = torch.clone(inpp)
                    
            inputs = inputs.float()
            if bina in ['sto', 'det'] :
                param_sauv = []
                for layer in model.layers :
                    if isinstance(layer, nn.Linear):
                        param_sauv.append(torch.clone(layer.weight.data))
                        if bina == 'det' :
                            layer.weight.data = torch.sign(layer.weight.data)
                        elif bina == 'sto' :
                            p = torch.clip((layer.weight.data+1)/2,0,1)
                            layer.weight.data = torch.sign(torch.bernoulli(p)-0.5)
            if str(DEVICE) == 'cuda' :
                inputs = inputs.cuda()
                targets = targets.cuda()
                model = model.cuda()
            output = model(inputs)
            if num_class != 1 :
                predictions = output.max(dim=1)[1]
            else :
                predictions = output
            val_loss.append(criterion(output, targets).item())
            true += targets.cpu().numpy().tolist()
            pred += predictions.cpu().numpy().tolist()
            inp += inputs.cpu().numpy().tolist()
            if bina in ['sto', 'det'] :
                j = 0
                for layer in model.layers :
                    if isinstance(layer, nn.Linear):
                        layer.weight.data = torch.clone(param_sauv[j])
                        j += 1
    if num_class == 2 :
        tru = []
        for i in range(len(true)) :
            if true[i][0] == -1 :
                tru.append(1)
            else :
                tru.append(0)
        true = tru
        
    if return_data == True :
        return true, pred, inp
    elif num_class == 1 :
        return 0, sum(val_loss) / len(val_loss)
    else :
        return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)


def test(model, job, algo, bina, dataset, batch_size, num_class, criterion, DEVICE, use_gpu=False):
    """
    Test un réseau de neurones de classification pour un certain nombre d'epochs 
    avec PyTorch. La fonction affiche l'exactitude et la perte moyenne.
    
    Args:
        model (nn.Module): Un réseau de neurones instancié avec PyTorch.
        dataset (Dataset): Un jeu de données PyTorch.
        batch_size (int): La taille des batchs.
        use_gpu (bool): Si les données doivent être envoyées sur GPU.
    """
    test_loader = DataLoader(dataset, batch_size=batch_size)
    test_acc, test_loss = validate(model, job, algo, bina, test_loader, num_class, criterion, DEVICE, use_gpu=use_gpu)
    print('Test acc: {:.2f} - Test loss: {:.4f}'.format(test_acc, test_loss))
    return test_acc, test_loss


def lets_go(experiment_name, job, method, dataset, random_seed, algo, bina, switch_train_test,
            loss_func, num_neur, num_hid_lay, network, optim_algo, nb_epoch, batch_size,
            start_lr, end_lr=False, factor=False, dec_lr=False, patience=False,
            init=False, early_stop=False, early_stop_value=0.01, momentum=0,
            nesterov=False, reg_type=None, lambd=0, beta=None, T=None):
    """
    Entraine un réseau de neurone sur un dataset présélectionné, avec une 
        configuration arbitraire, avec une fonction de perte sélectionnée, etc.
    
    Args:
        dataset (str): Un jeu de données PyTorch à charger; choix entre :
                        -partial_mnist_17
                        -partial_mnist_56
                        -full_mnist
                        
        algo (str): Type d'entrainement du réseau'; choix entre :
                        -bc (BinaryConnect)
                        -ebpb (Expectation Backpropagation, binary weights)
                        
        bina (str): Type de réseau à entrainer; choix entre :
                        -baseline
                        -sto (binary connect; stochastic)
                        -det (binary connect; deterministic)
                        
        switch_train_test (bool): Décide d'inverser ou non le train set et le test
                                    set, à des fins de rapidité d'entrainement.
                        
        loss_func (str): Une fonction de perte pour l'entrainement du réseau; choix entre :
                            -linear_loss
                            -multi_linear_loss
                            -cross_entropy_loss
                            -multi_margin_loss_2
                            -nllloss
                            
        network (str): Une configuration de réseau de neurones préexistante; choix entre :
                            -MnistNet
                            -DeepMnistNet
                            
        num_neur (int): Nombre de neurones par couche cachée dans le réseau
                            
        optim_algo (str): Algorithme d'optimisation du réseau; choix entre :
                            -sgd
                            -adagrad
                            -adam
        
        momentum (float): Momentum (pour optim_algo == sgd); défaut : 0.
                            
        nb_epoch (int): Nombre d'epoch pour l'entrainement.
        
        batch_size (int): Nombre d'observation par batch pour l'entrainement.
        
        start_lr (float): Taux d'apprentissage à la première epoch; si 'dec_lr'
                            est à False, alors ce sera le taux constant.
        
        end_lr (float): Taux d'apprentissage à la dernière epoch; mettre valeur 
                            si dec_lr == 'exponential'.
                            
        factor (float): Taux de décroissance du lr; mettre valeur si 
                            dec_lr == 'plateau'.
        
        dec_lr (str): Façon de décroitre du taux learning rate; choix entre :
                        -exponential
                        -plateau
                        
        init (str): Initialisation des poids dans le réseau. Si aucune valeur 
                    n'est donnée en entrée, initialisation par défaut de Python;
                    choix entre :
                        -'kaiming_unif'
                        -'kaiming_norm'
                        -'xavier_unif'
                        -'xavier_norm'
        
        early_stop (int): Le nombre d'epoch à attendre avant d'effectuer un 
                            early stopping
        
        early_stop_value (float): Valeur d'amélioration requise de la validation
                                    accuracy pour remettre à 0 le compteur du
                                    early stoppeur (en pourcentage d'accuracy;
                                    1 = 1%); par défaut, 0.01.
                                    
        nesterov (bool): Permet la variante de Nesterov du momentum. Doit être
                            à False si SGD n'est pas utilisé.
    """
    random.seed(random_seed + 0)
    torch.manual_seed(random_seed + 1)
    np.random.seed(random_seed + 2)
    cnt = 0
    try:
        with open("results_" + str(experiment_name) + ".txt", "r") as tes:
            tess = [line.strip().split('\t') for line in tes]
        tes.close()
        if algo in ['bc', 'bnn']:
            is_it_new = [str(switch_train_test), str(dataset[0]), str(random_seed),
                         str(algo), str(bina), str(loss_func), str(network),
                         str(num_neur), str(num_hid_lay), str(optim_algo), str(nb_epoch),
                         str(batch_size), str(start_lr), str(end_lr), str(factor),
                         str(dec_lr), str(patience), str(init), str(early_stop),
                         str(early_stop_value), str(momentum), str(nesterov), 
                         str(reg_type), str(lambd), str(beta), str(T)]
        if algo in ['ebpr', 'ebpb']:
            is_it_new = [str(switch_train_test), str(dataset[0]), str(random_seed),
                         str(algo), 'prob', '---', str(network),
                         str(num_neur), str(num_hid_lay), '---', str(nb_epoch),
                         str(batch_size), str(start_lr), '---', '---',
                         '---', '---', str(init), str(early_stop),
                         str(early_stop_value), '---', '---', '---', '---',
                         '---', '---']                        
        for a in tess:
            if a[0:len(is_it_new)] == is_it_new :
                cnt += 1
    except FileNotFoundError:
        file = open("results_" + str(experiment_name) + ".txt", "a")
        file.write(
            "switch_train_test\tdataset\ta_random_seed\talgo\tbina\tloss_func\tnetwork\tnum_neur\tnum_hid_lay\toptim_algo\tnb_epoch\tbatch_size\tstart_lr\tend_lr\tfactor\tdec_lr\tpatience\tinit\tearly_stop\tearly_stop_value\tmomentum\tnesterov\treg_type\tlambd\tbeta\tT\ttrain_acc\tvalid_acc\ttest_acc\ttest_loss\n")
        file.close()

        # Quelques tests afin de s'assurer que les entrées sont cohérentes les unes
    #   avec les autres.

    if (cnt >= 1):
        print("Already done; passing...\n")

    else:
        mnist, mnist_test = dataset[1]
        num_class = dataset[2]
        num_neur_ent = dataset[3]
        
        if switch_train_test == True:
            temp = mnist
            mnist = mnist_test
            mnist_test = temp

        if loss_func == 'linear_loss':
            criterion = linear_loss
        elif loss_func == 'cross_entropy_loss':
            criterion = nn.CrossEntropyLoss()
        elif loss_func == 'multi_margin_loss_2':
            criterion = nn.MultiMarginLoss(2)
        elif loss_func == 'nllloss':
            criterion = nn.NLLLoss()
            
        softmax = True
        BN = True
        if job in ['regression', 'pruning']:
            softmax = False
            
        if network == 'MnistNet':
            model = MnistNet(num_neur_ent, num_neur, num_class, num_hid_lay, algo, beta, T, nbatch=round(len(mnist) / batch_size), softmax=softmax, BN=BN)

        # Si un GPU est disponible, on fait appel à ce dernier; sinon, on entraine
        #   sur le CPU.    
        DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # DEVICE = torch.device('cpu')
        print(str(DEVICE) + " will be used.")
        if str(DEVICE) == 'cuda':
            model.cuda()
        print(model)

        # On entraine le modèle.
        if algo in ['bc', 'bnn']:
            # On initialise les poids.
            if init != False:
                init_weights(init, model, algo, mnist)
            
            model, bta, bva, _ = train(model, job, mnist, algo, bina, optim_algo, momentum, nb_epoch, batch_size,
                                    num_class, criterion, DEVICE, start_lr, end_lr, factor,
                                    dec_lr, patience, early_stop, early_stop_value, nesterov, reg_type, lambd, use_gpu=True,
                                    dat=dataset[1])
        
            test_acc, test_loss = test(model, job, algo, bina, mnist_test, batch_size, num_class, criterion, DEVICE,
                                       use_gpu=True)
            
            file = open("rslts/results_test_" + str(experiment_name) + '_' + str(job) + ".txt", "a")
            file.write(str(switch_train_test) + "\t" + str(dataset[0]) + "\t" + str(random_seed) + "\t" +
                       str(algo) + "\t" + str(bina) + "\t" + str(loss_func) + "\t" + str(network) + "\t" +
                       str(num_neur) + "\t" + str(num_hid_lay) + "\t" + str(optim_algo) + "\t" +
                       str(nb_epoch) + "\t" + str(batch_size) + "\t" + str(start_lr) + "\t" +
                       str(end_lr) + "\t" + str(factor) + "\t" +
                       str(dec_lr) + "\t" + str(patience) + "\t" + str(init) + "\t" +
                       str(early_stop) + "\t" + str(early_stop_value) + "\t" +
                       str(momentum) + "\t" + str(nesterov) + "\t" +
                       str(reg_type) + "\t" + str(lambd) + "\t" + str(beta) +
                       "\t" + str(T) + "\t" + str(bta) + "\t" + str(bva) + "\t" +
                       str(test_acc) + "\t" + str(test_loss) + "\n")
            file.close()


def main(experiment_name='test',
         task=['classification'],                                          
         method=[''],
         dataset=['partial_mnist_56',
                  'partial_mnist_49',
                  'partial_mnist_17'],
         random_seed=[0],
         algo=['bnn', 'bc'],
         bina=['sto'],                                             
         switch_train_test=[False],
         loss_func=['linear_loss'],
         network=['MnistNet'],
         num_neur=[100],
         num_hid_lay=[2],
         optim_algo=['adam'],
         momentum=[0],
         nb_epoch=[100],
         batch_size=[64],
         start_lr=[0.1, 0.01, 0.001],
         end_lr=[1],
         factor=[0.5],
         dec_lr=['plateau'],
         patience=[5],
         init=['kaiming_unif'],
         early_stop=[20],
         early_stop_value=[0.01],
         nesterov=[False]):
    param_grid = ParameterGrid([{'task': task,
                                 'method': method,
                                 'dataset': dataset,
                                 'a_random_seed': random_seed,
                                 'algo': algo,
                                 'bina': bina,
                                 'switch_train_test': switch_train_test,
                                 'loss_func': loss_func,
                                 'network': network,
                                 'num_neur': num_neur,
                                 'num_hid_lay': num_hid_lay,
                                 'optim_algo': optim_algo,
                                 'momentum': momentum,
                                 'nb_epoch': nb_epoch,
                                 'batch_size': batch_size,
                                 'start_lr': start_lr,
                                 'end_lr': end_lr,
                                 'factor': factor,
                                 'dec_lr': dec_lr,
                                 'patience': patience,
                                 'init': init,
                                 'early_stop': early_stop,
                                 'early_stop_value': early_stop_value,
                                 'nesterov': nesterov}])
    param_grid = [t for t in param_grid]
    ordering = {d: i for i, d in enumerate(dataset)}
    param_grid = sorted(param_grid, key=lambda d: ordering[d['dataset']])
    n_tasks = len(param_grid)

    last_loaded_dataset = ''
    last_seed = ''

    for n in range(1):
        for i, task_dict in enumerate(param_grid):
            print(f"Launching task {n_tasks * n + i + 1}/{n_tasks * 1} : {task_dict}")

            if task_dict['dataset'] == 'partial_mnist_17' and (
                    last_loaded_dataset != 'partial_mnist_17' or last_seed != task_dict['a_random_seed']):
                data = 'partial_mnist_17', load_mnist(1, 7, task_dict['a_random_seed']), 2, 28**2
                last_loaded_dataset = 'partial_mnist_17'
            
            elif task_dict['dataset'] == 'partial_mnist_56' and (
                    last_loaded_dataset != 'partial_mnist_56' or last_seed != task_dict['a_random_seed']):
                data = 'partial_mnist_56', load_mnist(5, 6, task_dict['a_random_seed']), 2, 28 ** 2
                last_loaded_dataset = 'partial_mnist_56'
            
            elif task_dict['dataset'] == 'partial_mnist_49' and (
                    last_loaded_dataset != 'partial_mnist_49' or last_seed != task_dict['a_random_seed']):
                data = 'partial_mnist_49', load_mnist(4, 9, task_dict['a_random_seed']), 2, 28**2
                last_loaded_dataset = 'partial_mnist_49'
            
            last_seed = task_dict['a_random_seed']
            
            if task_dict['dataset'] in ['partial_mnist_17', 'partial_mnist_49', 'partial_mnist_56', 'mnist_lh'] :
                job = 'classification'
            lets_go(experiment_name=experiment_name,
                    job=job,
                    method=method,
                    dataset=data,
                    random_seed=task_dict['a_random_seed'],
                    algo=task_dict['algo'],
                    bina=task_dict['bina'],
                    switch_train_test=task_dict['switch_train_test'],
                    loss_func=task_dict['loss_func'],
                    network=task_dict['network'],
                    num_neur=task_dict['num_neur'],
                    num_hid_lay=task_dict['num_hid_lay'],
                    optim_algo=task_dict['optim_algo'],
                    momentum=task_dict['momentum'],
                    nb_epoch=task_dict['nb_epoch'],
                    batch_size=task_dict['batch_size'],
                    start_lr=task_dict['start_lr'],
                    end_lr=task_dict['end_lr'],
                    factor=task_dict['factor'],
                    dec_lr=task_dict['dec_lr'],
                    patience=task_dict['patience'],
                    init=task_dict['init'],
                    early_stop=task_dict['early_stop'],
                    early_stop_value=task_dict['early_stop_value'],
                    nesterov=task_dict['nesterov'])

    print("### ALL TASKS DONE ###")


if __name__ == '__main__':
    main()
