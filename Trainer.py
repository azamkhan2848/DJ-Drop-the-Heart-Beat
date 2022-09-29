import os
import matplotlib.pyplot as plt
from typing import Callable, Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import torch.nn.functional as F
from torch.utils.data import DataLoader


class ProgressBar:
    last_length = 0

    @staticmethod
    def show(prefix: str, postfix: str, current: int, total: int, newline: bool = False) -> None:
        progress = (current + 1) / total
        if current == total:
            progress = 1

        current_progress = progress * 100
        progress_bar = '=' * int(progress * 20)

        message = ''

        if len(prefix) > 0:
            message += f'{prefix}, [{progress_bar:<20}]'

            if not newline:
                message += f' {current_progress:6.2f}%'

        if len(postfix) > 0:
            message += f', {postfix}'

        print(f'\r{" " * ProgressBar.last_length}', end='')
        print(f'\r{message}', end='')

        if newline:
            print()
            ProgressBar.last_length = 0
        else:
            ProgressBar.last_length = len(message) + 1


class BaseTrainer:
    def __init__(self) -> None:
        self.device = 'cuda:0' if cuda.is_available() else 'cpu'

    def train(self) -> None:
        raise NotImplementedError('train not implemented')

    def test(self) -> None:
        raise NotImplementedError('test not implemented')

    @property
    def weights(self) -> None:
        raise NotImplementedError('weights not implemented')


class IEGMTrainer(BaseTrainer):
    def __init__(self, net: nn.Module, optimizer: optim.Optimizer, criterion: Callable, model_name) -> None:
        super(IEGMTrainer, self).__init__()
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_name = model_name

        self.net = self.net.to(self.device)

    def _step(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device)

        outputs = self.net(x)

        running_acc = (outputs.argmax(1) == y).type(torch.float).sum()

        running_loss = self.criterion(outputs, y)

        return running_loss, running_acc

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader = None, scheduler: Any = None) -> None:
        epoch_length = len(str(epochs))

        stale = 0

        best_acc = 0.0

        for epoch in range(epochs):
            self.net.train()

            loss = 0.0
            acc = 0.0

            for i, data in enumerate(train_loader):
                inputs, labels = data['IEGM_seg'], data['label']

                self.optimizer.zero_grad()

                running_loss, running_acc = self._step(x=inputs, y=labels)

                running_loss.backward()
                self.optimizer.step()

                loss += running_loss.item()
                acc += running_acc.item()

                prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                postfix = f'loss: {running_loss.item():.3f}'
                ProgressBar.show(prefix, postfix, i, len(train_loader))

            loss /= len(train_loader)
            acc /= len(train_loader.dataset)

            prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
            postfix = f'loss: {loss:.3f}, acc: {acc:.3f}'
            ProgressBar.show(prefix, postfix, len(train_loader), len(train_loader), newline=True)

            if val_loader:
                val_loss, val_acc = self.test(val_loader)

                if val_acc > best_acc:
                    print("Best model found at epoch {}".format(epoch+1))
                    # torch.save(self.net.state_dict(), f"./saved_models/{self.model_name}.ckpt")
                    torch.save(self.net, f'./saved_models/{self.model_name}.pkl')
                    # torch.save(self.net.state_dict(), f'./saved_models/{self.model_name}_state_dict.pkl')
                    best_acc = val_acc
                    stale = 0
                else:
                    stale += 1
                    if stale > 20:
                        break

            if scheduler:
                scheduler.step()
            
    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> None:
        self.net.eval()

        loss = 0.0
        acc = 0.0

        for i, data in enumerate(test_loader):
            inputs, labels= data['IEGM_seg'], data['label']
            running_loss, running_acc = self._step(x=inputs, y=labels)

            loss += running_loss.item()
            acc += running_acc.item()

            prefix = 'Test'
            postfix = f'loss: {running_loss.item():.3f}'
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        loss /= len(test_loader)
        acc /= len(test_loader.dataset)

        prefix = 'Test'
        postfix = f'loss: {loss:.3f}, acc: {acc:.3f}'
        ProgressBar.show(prefix, postfix, len(test_loader), len(test_loader), newline=True)

        return loss, acc

    @property
    @torch.no_grad()
    def weights(self) -> dict:
        return {'net': self.net}

class IEGM_kd_Trainer(BaseTrainer):
    def __init__(self, teacher: nn.Module, student: nn.Module, optimizer: optim.Optimizer, model_name, teacher_name) -> None:
        super(IEGM_kd_Trainer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.model_name = model_name

        self.teacher = self.teacher.to(self.device)
        self.student = self.student.to(self.device)
        
    def loss_fn_kd(self, student_out, labels, teacher_out):
        alpha = 0.9606297333554856
        T = 818.5522883592068
        kd_loss = nn.KLDivLoss()(F.log_softmax(student_out/T, dim=1), 
                                 F.softmax(teacher_out/T, dim=1))*(alpha*T*T)+ \
                                F.cross_entropy(student_out, labels)*(1.-alpha)
        return kd_loss

    def _step(self, x: torch.Tensor, y: torch.tensor, train=True) -> torch.Tensor:
        x = x.to(self.device, dtype=torch.float)
        y = y.to(self.device)

        teacher_out = self.teacher(x)

        student_out = self.student(x)

        running_acc = (student_out.argmax(1) == y).type(torch.float).sum()

        if train:
            running_loss = self.loss_fn_kd(student_out, y, teacher_out)
        else:
            running_loss = nn.CrossEntropyLoss()(student_out, y)

        return running_loss, running_acc

    def train(self, epochs: int, train_loader: DataLoader, val_loader: DataLoader = None, scheduler: Any = None) -> None:
        epoch_length = len(str(epochs))

        stale = 0

        best_acc = 0.0

        self.teacher.eval()

        for epoch in range(epochs):
            self.student.train()

            loss = 0.0
            acc = 0.0

            for i, data in enumerate(train_loader):
                inputs, labels = data['IEGM_seg'], data['label']

                self.optimizer.zero_grad()

                running_loss, running_acc = self._step(x=inputs, y=labels)

                running_loss.backward()
                self.optimizer.step()

                loss += running_loss.item()
                acc += running_acc.item()

                prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
                postfix = f'loss: {running_loss.item():.3f}'
                ProgressBar.show(prefix, postfix, i, len(train_loader))

            loss /= len(train_loader)
            acc /= len(train_loader.dataset)

            prefix = f'Epochs: {(epoch + 1):>{epoch_length}} / {epochs}'
            postfix = f'loss: {loss:.3f}, acc: {acc:.3f}'
            ProgressBar.show(prefix, postfix, len(train_loader), len(train_loader), newline=True)

            if val_loader:
                val_loss, val_acc = self.test(val_loader)

                if val_acc > best_acc:
                    print("Best model found at epoch {}".format(epoch+1))
                    torch.save(self.student, f'./saved_models/{self.model_name}.pkl')
                    # torch.save(self.student.state_dict(), f'./saved_models/{self.model_name}_state_dict.pkl')
                    best_acc = val_acc
                    stale = 0
                else:
                    stale += 1
                    if stale > 20:
                        break

            if scheduler:
                scheduler.step()
            
    @torch.no_grad()
    def test(self, test_loader: DataLoader) -> None:
        self.student.eval()

        loss = 0.0
        acc = 0.0

        for i, data in enumerate(test_loader):
            inputs, labels= data['IEGM_seg'], data['label']
            running_loss, running_acc = self._step(x=inputs, y=labels, train=False)

            loss += running_loss.item()
            acc += running_acc.item()

            prefix = 'Test'
            postfix = f'loss: {running_loss.item():.3f}'
            ProgressBar.show(prefix, postfix, i, len(test_loader))

        loss /= len(test_loader)
        acc /= len(test_loader.dataset)

        prefix = 'Test'
        postfix = f'loss: {loss:.3f}, acc: {acc:.3f}'
        ProgressBar.show(prefix, postfix, len(test_loader), len(test_loader), newline=True)

        return loss, acc

    @property
    @torch.no_grad()
    def weights(self) -> dict:
        return {'net': self.net}