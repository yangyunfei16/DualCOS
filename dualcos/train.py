from __future__ import print_function

import json
import os
import random
import numpy as np
from pprint import pprint
import copy
from torchvision import transforms

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloader
from cli_parser import parse_args
from metrics import eval_and_log_metrics, eval_and_log_validation_metrics, log_hparams, log_generator_distribution
from my_utils import *
from approximate_gradients import estimate_gradient_objective

from teacher import TeacherModel
from ensemble import Ensemble
from replay import init_replay_memory
import config
import time

# tensor to image
toPIL = transforms.ToPILImage()

# generator training process
def train_generator(args, generator, student_ensemble, teacher, device, optimizer, targets, j): # generator training
    """Train generator model. Methodology is based on cli input args, especially the experiment-type parameter."""
    assert not teacher.training
    generator.train()
    student_ensemble.eval()

    g_loss_sum = 0
    for i in range(args.g_iter):
        optimizer.zero_grad()
        z = torch.randn((args.batch_size, args.nz)).to(device)
        # z = torch.randn(2, 1, 512).to(device).unbind(0)  # add code
        if args.experiment_type == 'dfme':
            g_loss = dfme_gen_loss(args, z, generator, student_ensemble, teacher, device)
        elif args.experiment_type == 'dualcos':
            g_loss = dualcos_gen_loss(z, generator, student_ensemble, args, targets, j) # modify code
        optimizer.step()
        g_loss_sum += g_loss
    return g_loss_sum / args.g_iter


def dfme_gen_loss(args, z, generator, student_ensemble, teacher, device):
    """Compute the generator loss for DFME method. Uses forward differences method. Update weights based on loss.
    See Also: https://github.com/cake-lab/datafree-model-extraction"""
    fake = generator(z, pre_x=args.approx_grad)  # pre_x returns the output of G before applying the activation

    # Estimate gradient for black box teacher
    approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, teacher, student_ensemble, fake,
                                                            epsilon=args.grad_epsilon, m=args.grad_m,
                                                            device=device, pre_x=True)

    fake.backward(approx_grad_wrt_x)
    return loss_G.item()


# Label smoothing
class ScoreLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(ScoreLoss, self).__init__()
        self.reduction = reduction

    def forward(self, logits, target):
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), logits.size(1), -1)  # [N, C, HW]
            logits = logits.transpose(1, 2)  # [N, HW, C]
            logits = logits.contiguous().view(-1, logits.size(2))  # [NHW, C]
        target = target.view(-1, 1)  # [NHW，1]

        score = F.log_softmax(logits, 1)  # score-based
        score = score.gather(1, target)  # [NHW, 1]
        loss = -1 * score

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
score_loss = ScoreLoss()

def dualcos_gen_loss(z, generator, student_ensemble, args, targets, j): # modify code
    """Compute generator loss for DualCOS method. Update weights based on loss.
    Calculates weighted average between prediction discrepancy loss and class diversity loss."""
    fake = generator(z)

    preds = []
    # student_ensemble_size = args.ensemble_size
    # print(student_ensemble_size)
    for idx in range(student_ensemble.size()):
    # for idx in range(student_ensemble_size):  # add code
        preds.append(student_ensemble(fake, idx=idx))  # 2x [batch_size, K] Last dim is logits
    
    # add code
    # loss_G4_1 = 0.5*score_loss(preds[0], targets) + 0.5*score_loss(preds[1], targets)

    # loss_G2 = -F.l1_loss(preds[0], preds[1])            # add code
    # loss_G2 = -F.cross_entropy(preds[0], preds[1])            # add code
    # loss_G2 = -F.kl_div(preds[0].softmax(dim=-1).log(), preds[1].softmax(dim=-1), reduction="batchmean")   # add code
    loss_G2 = -F.l1_loss(preds[0].softmax(dim=-1), preds[1].softmax(dim=-1))            # prediction discrepancy loss
    # loss_G4 = F.cross_entropy(preds[0], targets) + F.cross_entropy(preds[1], targets)          # add code
    preds = torch.stack(preds, dim=1)                  # [batch_size, 2, K]
    preds = F.softmax(preds, dim=2)                    # [batch_size, 2, K] Last dim is confidence values.
    # loss_G4 = F.cross_entropy(torch.mean(preds, dim=1), targets)  # add code

    # pseudo label-guided loss
    smoothing = 0.03 # add code
    confidence = 1.0 - smoothing  # add code
    logprobs = torch.log(torch.mean(preds, dim=1)) # add code
    nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -logprobs.mean(dim=-1)
    loss_G4 = confidence * nll_loss + smoothing * smooth_loss
    loss_G4 = loss_G4.mean()                    # add code
    
    std = torch.std(preds, dim=1)                      # std has shape [batch_size, K]. standard deviation over models
    loss_G = -torch.mean(std)                          # original Disagreement Loss
    # loss_G4 = 0.5*loss_G4                    # add code
    loss_G = loss_G2  # add code
    if args.lambda_div != 0:
        soft_vote_mean = torch.mean(torch.mean(preds + 0.000001, dim=1),
                                    dim=0)  # [batch_size, 2, K] -> [batch_size, K] -> [K]
        loss_G += args.lambda_div * (torch.sum(soft_vote_mean * torch.log(soft_vote_mean)))  # Diversity Loss
        # if j < 5: # add code
        loss_G = loss_G + 0.01*loss_G4  # add code
    # loss_G = loss_G + 0.00*loss_G4  # add code
    loss_G.backward()
    return loss_G.item()


def supervised_student_training(student_ensemble, fake, t_logit, optimizer, args):
    """Calculate loss and update weights for students in a supervised fashion"""
    student_iter_preds = []
    student_iter_loss = 0
    for i in range(student_ensemble.size()):
        s_logit = student_ensemble(fake, idx=i)
        with torch.no_grad():
            student_iter_preds.append(F.softmax(s_logit, dim=-1).detach())
        loss_s = student_loss(s_logit, t_logit, args)  # Helper function which handles soft- and hard-label settings

        # Boundary Sample Weighted Training
        idx = torch.where(s_logit.max(1)[1] != t_logit.max(1)[1])[0]
        loss_adv = F.cross_entropy(s_logit[idx], t_logit[idx].max(1)[1])
        loss_s = loss_s + 0.01*loss_adv

        loss_s.backward()
        student_iter_loss += loss_s.item()
    optimizer.step()
    return torch.stack(student_iter_preds, dim=1), student_iter_loss


# Semi-supervised training function
def semi_supervised_student_training(student_ensemble, fake, t_logit, fake_new, optimizer, args):
    """Calculate loss and update weights for students in a supervised fashion"""
    student_iter_preds = []
    student_iter_loss = 0

    for i in range(student_ensemble.size()):
        s_logit = student_ensemble(fake, idx=i)
        with torch.no_grad():
            student_iter_preds.append(F.softmax(s_logit, dim=-1).detach())
        loss_s = student_loss(s_logit, t_logit, args)  # Helper function which handles soft- and hard-label settings

        preds = student_ensemble(fake_new, idx=i)  # 2x [batch_size, K] Last dim is logits
        preds = F.softmax(preds, dim=1)
        soft_vote_mean = torch.mean(preds, dim=0)
        loss_e = -1 * (torch.sum(soft_vote_mean * torch.log(soft_vote_mean)))
        loss_s = loss_s + 0.02*loss_e

        loss_s.backward()
        student_iter_loss += loss_s.item()
    optimizer.step()
    return torch.stack(student_iter_preds, dim=1), student_iter_loss


# Sampling process based on active learning
def select_fake(student_ensemble, fake, device, args):
    preds = []
    for idx in range(student_ensemble.size()):
        preds.append(student_ensemble(fake, idx=idx))  # 2x [batch_size, K] Last dim is logits
    
    preds = torch.stack(preds, dim=1)                  # [batch_size, 2, K]
    preds = F.softmax(preds, dim=2)                    # [batch_size, 2, K] Last dim is confidence values.
    soft_vote_mean = torch.mean(preds + 0.000001, dim=1)  # [batch_size, 2, K] -> [batch_size, K]

    entropy = []
    # print(soft_vote_mean)
    for idx in range(soft_vote_mean.size(0)):
        # print(soft_vote_mean[idx])
        entropy.append((-torch.sum(soft_vote_mean[idx] * torch.log(soft_vote_mean[idx]))).item())  # Diversity Loss
        # print(entropy)
    t = copy.deepcopy(entropy)
    max_number = []
    max_index = []
    for _ in range(args.batch_size):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_number.append(number)
        max_index.append(index)
    t = []
    # print(max_number)
    # print(max_index)
    indices = torch.tensor(max_index).to(device)
    select_data = torch.index_select(fake, 0, indices)
    # print(select_data.size())

    return select_data

# clone model training process
def train_student_ensemble(args, generator, student_ensemble, teacher, device, optimizer, replay_memory):
    """Train student ensemble with a fixed generator"""
    assert not teacher.training
    generator.eval()
    student_ensemble.train()

    s_loss_sum = 0
    student_preds = []
    teacher_preds = []
    for d_iter in range(args.d_iter):  # Generate and train for d_iter batches. Store batches to experience replay
        optimizer.zero_grad()
        # z = torch.randn((args.batch_size, args.nz)).to(device)  # Sample from random number generator
        z = torch.randn((500, args.nz)).to(device)  # add code
        # z = torch.randn((2, 1, 512).unbind(0)).to(device)  # add code
        fake = generator(z).detach()                            # Generate synthetic data with generator
        fake = select_fake(student_ensemble, fake, device, args) # add code
        t_logit = get_teacher_prediction(args, teacher, fake)   # Query teacher model and update query budget
        replay_memory.update(fake.cpu(), t_logit.cpu())         # Store queries to experience replay
        student_iter_preds, student_iter_loss = supervised_student_training(student_ensemble, fake, t_logit,
                                                                            optimizer, args)  # Train students

        teacher_preds.append(F.softmax(t_logit, dim=-1).detach())  # Store teacher predictions for logging purposes
        student_preds.append(student_iter_preds)                   # Store student predictions for logging purposes
        s_loss_sum += student_iter_loss                            # Store student loss for logging purposes

    for _ in range(args.rep_iter):  # Sample buffer reuse. Train for rep_iter batches on samples from experience replay.
        optimizer.zero_grad()
        fake, t_logit = replay_memory.sample()  # Sample features and labels from experience replay.  # add code
        fake.to(device)                         # Load features to device
        t_logit.to(device)                      # Load labels/teacher predictions to device
        student_iter_preds, student_iter_loss = supervised_student_training(student_ensemble, fake, t_logit,
                                                                                optimizer, args)  # Train students
        teacher_preds.append(F.softmax(t_logit, dim=-1).detach())  # Store teacher predictions for logging purposes
        student_preds.append(student_iter_preds)                   # Store student predictions for logging purposes
        s_loss_sum += student_iter_loss                            # Store student loss for logging purposes

    # Prep student and teacher preds for logging purposes
    teacher_preds = torch.cat(teacher_preds, dim=0)
    student_preds = torch.cat(student_preds, dim=0)

    return student_preds, teacher_preds, s_loss_sum / (args.d_iter * student_ensemble.size())


# clone model training process during Semi-supervised boosting stage
def train_student_ensemble_boost(args, student_ensemble, generator, device, optimizer, epoch, replay_memory, selected_idx):
    """Train student ensemble with a fixed generator"""
    # assert not teacher.training
    # generator.eval()
    student_ensemble.train()

    s_loss_sum = 0
    student_preds = []
    teacher_preds = []
    # for d_iter in range(args.d_iter):  # Generate and train for d_iter batches. Store batches to experience replay
    #     optimizer.zero_grad()
    #     # z = torch.randn((args.batch_size, args.nz)).to(device)  # Sample from random number generator
    #     z = torch.randn((400, args.nz)).to(device)  # add code
    #     fake = generator(z).detach()                            # Generate synthetic data with generator
    #     fake = select_fake(student_ensemble, fake, device, args) # add code
    #     t_logit = get_teacher_prediction(args, teacher, fake)   # Query teacher model and update query budget
    #     replay_memory.update(fake.cpu(), t_logit.cpu())         # Store queries to experience replay
    #     student_iter_preds, student_iter_loss = supervised_student_training(student_ensemble, fake, t_logit,
    #                                                                         optimizer, args)  # Train students

    #     teacher_preds.append(F.softmax(t_logit, dim=-1).detach())  # Store teacher predictions for logging purposes
    #     student_preds.append(student_iter_preds)                   # Store student predictions for logging purposes
    #     s_loss_sum += student_iter_loss                            # Store student loss for logging purposes

    for i in range(1, args.rep_iter + 98):  # Train for rep_iter batches on samples from experience replay.
        optimizer.zero_grad()
        fake, t_logit = replay_memory.sample_boost(selected_idx)  # Sample features and labels from experience replay.  # add code
        fake.to(device)                         # Load features to device
        t_logit.to(device)                      # Load labels/teacher predictions to device

        # idx_s = perm(len(fake_s), args.batch_size, device).cpu()
        # # print(idx_s)
        # fake_new = fake_s[idx_s].to(device)
        # # print(fake_new)

        z_s = torch.randn((args.batch_size, args.nz)).to(args.device)  # add code
        fake_s = generator(z_s).detach()                            # Generate synthetic data with generator

        student_iter_preds, student_iter_loss = semi_supervised_student_training(student_ensemble, fake, t_logit,
                                                                                fake_s, optimizer, args)  # Train students
        teacher_preds.append(F.softmax(t_logit, dim=-1).detach())  # Store teacher predictions for logging purposes
        student_preds.append(student_iter_preds)                   # Store student predictions for logging purposes
        s_loss_sum += student_iter_loss                            # Store student loss for logging purposes

        # replay_memory.sample_delete(idx)
        if i % 20 == 0:
            s_loss = s_loss_sum / (20*student_ensemble.size())
            print_and_log(f'Train Epoch: {epoch} [{i}/{args.rep_iter + 97} ({100*float(i)/float(args.rep_iter + 97):.0f}%)] '
                  f'Student Loss:{s_loss}')
            s_loss_sum = 0
        # log_training_epoch_boost(args, s_loss, i, epoch)  # Command line logging of training state.

    # Prep student and teacher preds for logging purposes
    teacher_preds = torch.cat(teacher_preds, dim=0)
    student_preds = torch.cat(student_preds, dim=0)

    return student_preds, teacher_preds, s_loss_sum / (args.d_iter * student_ensemble.size())


def log_training_epoch(args, g_loss, s_loss, i, epoch):
    """Somewhat dated function for logging training data to command line and logfile"""
    if i % args.log_interval != 0:
        return
    print_and_log(f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100*float(i)/float(args.epoch_itrs):.0f}%)] '
                  f'Generator Loss:{g_loss} Student Loss:{s_loss}')

# add code
def log_training_epoch_boost(args, s_loss, i, epoch):
    """Somewhat dated function for logging training data to command line and logfile"""
    if i % 10 != 0:
        return
    print_and_log(f'Train Epoch: {epoch} [{i}/{args.rep_iter + 47} ({100*float(i)/float(args.rep_iter + 47):.0f}%)] '
                  f'Student Loss:{s_loss}')

# generator resetting
def reset_generator(model):
    for m in model.modules():
        if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)


def train_epoch_ensemble(args, generator, student_ensemble, teacher, device,
                         optimizer_student, optimizer_generator, epoch, replay_memory):
    """Runs alternating generator and student training iterations.
    Also verifies queries counts match up with theoretically expected values."""
    student_preds = None
    teacher_preds = None
    # reset_generator(generator)  # add code

    for i in range(args.epoch_itrs):

        # Generate pseudo labels
        targets = torch.tensor([i for i in range(10)])
        list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        slice = random.sample(list, 6)
        targets_1 = torch.tensor([i for i in slice])
        targets = torch.cat((targets.repeat(int(256 / 10)), targets_1), dim=0)
        targets = targets.to(args.device)

        g_loss = train_generator(args, generator, student_ensemble, teacher, device, optimizer_generator, targets, i) # modify code
        student_preds, teacher_preds, s_loss = train_student_ensemble(args, generator, student_ensemble, teacher,
                                                                      device, optimizer_student, replay_memory)
        
        # Update query budget based on theoretically expected values. Then verify it matches up with actual value.
        args.query_budget -= args.cost_per_iteration
        assert (args.query_budget + args.current_query_count == args.total_query_budget), f"{args.query_budget} + {args.current_query_count}"

        log_training_epoch(args, g_loss, s_loss, i, epoch)  # Command line logging of training state.
        if args.query_budget < args.cost_per_iteration:  # End training if we cannot complete full iteration
            break
    return student_preds, teacher_preds


def log_test_metrics(model, test_loader, teacher_test_preds, device, args, task):
    if not isinstance(model, Ensemble):
        print_and_log(f"log_test_metrics currently only supports Ensemble. Detected class:{model.__class__}")
        raise NotImplementedError

    preds, labels = get_model_preds_and_true_labels(model, test_loader, device)
    stats = eval_and_log_metrics(config.tboard_writer, preds, labels, args, task)
    print_and_log(
        'Accuracies=> Soft Vote:{:.4f}%, Hard Vote:{:.4f}%, Es Median/Min/Max:{:.4f}%/{:.4f}%/{:.4f}%\n'.format(
            100 * stats["Soft Vote"]["Accuracy"],
            100 * stats["Hard Vote"]["Accuracy"],
            100 * stats["Ensemble"]["Accuracy"]["Median"],
            100 * stats["Ensemble"]["Accuracy"]["Min"],
            100 * stats["Ensemble"]["Accuracy"]["Max"]))
    eval_and_log_validation_metrics(config.tboard_writer, preds, teacher_test_preds, args, task, tag="Fidelity")
    return stats["Soft Vote"]["Accuracy"]


def get_model_accuracy_and_loss(args, model, loader, device="cuda"):
    """Get model accuracy and loss. Simple function intended for simply CLI printing. Prefer using metrics.py"""
    model.eval()
    correct, loss = 0, 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch
    loss /= len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return accuracy, loss


def init_logs(args):
    """Init log files. Mostly legacy behavior from DFME codebase."""
    os.makedirs(args.log_dir, exist_ok=True)

    # Save JSON with parameters
    with open(args.log_dir + "/parameters.json", "w") as f:
        json.dump(vars(args), f)

    init_or_append_to_log_file(args.log_dir, "loss.csv", "epoch,loss_G,loss_S")
    init_or_append_to_log_file(args.log_dir, "accuracy.csv", "epoch,accuracy")
    init_or_append_to_log_file(os.getcwd(), "latest_experiments.txt",
                               args.experiment_name + ":" + args.log_dir, mode="a")

    if args.rec_grad_norm:
        init_or_append_to_log_file(args.log_dir, "norm_grad.csv", "epoch,G_grad_norm,S_grad_norm,grad_wrt_X")


def main():
    # Parse command line arguments and set certain variables based on those
    args = parse_args(set_total_query_budget=True)
    # Print log directory
    print(args.log_dir)

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = False

    # Load test_loader and transform functions based on CLI args. Ignore train_loader.
    _, test_loader, normalization = get_dataloader(args)
    print(f"\nLoaded {args.dataset} successfully.")
    # Display distribution information of dataset
    print_dataset_feature_distribution(data_loader=test_loader)

    # Initialize log files and directories
    init_logs(args)
    args.model_dir = f"{args.experiment_dir}/student_{args.model_id}"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(f"{args.model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    config.log_file = open(f"{args.model_dir}/logs.txt", "w")

    # Set compute device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    print(f"Device is {args.device}")

    # Initialize tensorboard logger. Metrics are handled in metrics.py
    config.tboard_writer = SummaryWriter(f"tboard/general/{args.experiment_name}")

    args.normalization_coefs = None
    args.G_activation = torch.tanh

    # Set default number of classes. This will be moved to the CLI parser, eventually
    num_classes = 10 if args.dataset in ['cifar10', 'svhn', 'mnist'] else 100
    num_channels = 1 if args.dataset in ['mnist'] else 3
    args.num_classes = num_classes

    pprint(args, width=80)

    # Init teacher
    if args.model == 'resnet34_8x':
        teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)

        args.ckpt = 'checkpoint/teacher/'+ args.dataset +'-resnet34_8x.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    elif args.model =='resnet18_8x':
        teacher = network.resnet_8x.ResNet18_8x(num_classes=num_classes)

        args.ckpt = 'checkpoint/teacher/'+ args.dataset +'-resnet18_8x.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    elif args.model == 'lenet5':
        teacher = network.lenet.LeNet5()

        args.ckpt = 'checkpoint/teacher/'+ args.dataset +'-lenet5.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    else:
        teacher = get_classifier(args.model, pretrained=True, num_classes=args.num_classes)

    # Wrap teacher model in a handler class together with the data transform
    teacher = TeacherModel(teacher, transform=normalization)
    teacher.eval()
    config.tboard_writer.add_graph(teacher, torch.rand((32, num_channels, 32, 32)))
    teacher = teacher.to(args.device)

    # Evaluate teacher on test dataset to verify accuracy is in expected range
    print_and_log("Teacher restored from %s" % (args.ckpt))
    print(f"\n\t\tTraining with {args.model} as a Target\n")
    accuracy, _ = get_model_accuracy_and_loss(args, teacher, test_loader, args.device)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(np.round(accuracy * len(test_loader.dataset)),
                                                                     len(test_loader.dataset), accuracy))

    # Initialize a fresh student ensemble. Ensemble may be of size 1.
    student = get_classifier(args.student_model, pretrained=False, num_classes=args.num_classes,
                             ensemble_size=args.ensemble_size)
    for i in range(args.ensemble_size):
        config.tboard_writer.add_graph(student.get_model_by_idx(i), torch.rand((32, num_channels, 32, 32)))
    student = student.to(args.device)

    # Initialize generator
    generator = network.gan.GeneratorA(nz=args.nz, nc=num_channels, img_size=32, activation=args.G_activation,
                                       grayscale=args.grayscale)
    # generator = network.stylegan2.Generator_StyleGAN2()
    # config.tboard_writer.add_graph(generator, torch.rand((32, args.nz)))
    generator = generator.to(args.device)

    # # Enable multiple GPUs for parallel training
    # generator = torch.nn.DataParallel(generator, device_ids=[3,4,5])
    # student = torch.nn.DataParallel(student, device_ids=[3,4,5])
    # teacher = torch.nn.DataParallel(teacher, device_ids=[3,4,5])
    # model_list = [generator, student, teacher]
    # for model in model_list:
    #     # status = 0
    #     if isinstance(model,torch.nn.DataParallel):
    #         # if model==student:
    #         #     print('I am here!!!')
    #         #     status = 1
    #         model = model.module
    #     # if status == 1:
    #     #     print(model)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    # Compute theoretical query cost per training iteration. This will be compared with true value to verify correctness
    args.cost_per_iteration = compute_cost_per_iteration(args)
    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    # Init optimizers for student and generator
    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    # Compute learning rate drop iterations based on input percentages and iteration count.
    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: {}\n".format(steps))

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    best_acc = 0
    best_acc_epoch = 0  # add code
    acc_list = []
    replay_memory = init_replay_memory(args)
    teacher_test_preds, _ = get_model_preds_and_true_labels(teacher, test_loader, args.device)

    # Accuracy milestones to log
    accuracy_goals = {0.75: 0, 0.8: 0, 0.85: 0, 0.9: 0}

    """Stage 1: Interactive Training"""
    # Outer training loop.
    for epoch in range(1, number_epochs + 1):
        print_and_log(f"{args.experiment_name} epoch {epoch}")

        if epoch % 20 == 0:
            print("current student's lr: ", optimizer_S.state_dict()['param_groups'][0]['lr'])

        config.tboard_writer.add_scalar('Param/student_learning_rate', scheduler_S.get_last_lr()[0], args.current_query_count)
        config.tboard_writer.add_scalar('Param/generator_learning_rate', scheduler_G.get_last_lr()[0],
                                 args.current_query_count)

        # Inner training loop call
        student_preds, teacher_preds = train_epoch_ensemble(args, generator, student, teacher, args.device,
                                                            optimizer_S, optimizer_G, epoch, replay_memory)
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()
        replay_memory.new_epoch()

        # Test and log
        acc = log_test_metrics(student, test_loader, teacher_test_preds, args.device, args, task="multiclass")
        for goal in accuracy_goals:
            if accuracy_goals[goal] == 0 and acc > goal:
                accuracy_goals[goal] = args.current_query_count / 1000000.0
                print_and_log(f"Reached {goal} accuracy goal in {accuracy_goals[goal]}m queries")

        eval_and_log_validation_metrics(config.tboard_writer, student_preds, teacher_preds, args, task="multiclass")
        log_generator_distribution(config.tboard_writer, teacher_preds, args)

        acc_list.append(acc)
        # Store models
        if acc > best_acc:
            if not os.path.exists(f"{args.experiment_dir}/model_checkpoints"):
                os.makedirs(f"{args.experiment_dir}/model_checkpoints")
            best_acc = acc
            best_acc_epoch = epoch   # add code
            torch.save(student.state_dict(), f"{args.experiment_dir}/model_checkpoints/student_best.pt")
            torch.save(generator.state_dict(), f"{args.experiment_dir}/model_checkpoints/generator_best.pt")
        if epoch % args.generator_store_frequency == 0:
            if not os.path.exists(f"{args.experiment_dir}/model_checkpoints/generators"):
                os.makedirs(f"{args.experiment_dir}/model_checkpoints/generators")
            torch.save(generator.state_dict(),
                       f"{args.experiment_dir}/model_checkpoints/generators/generator_{args.current_query_count}.pt")


    """Stage 2: Semi-Supervised Boosting"""
    # Semi-supervised boosting loop.  # add code
    print("\n")
    print("==================================================================")
    print("Semi-supervised boosting loop start...")
    print("==================================================================")
    print("\n")

    optimizer_S = optim.SGD(student.parameters(), lr=0.003, weight_decay=args.weight_decay, momentum=0.9)
    steps = sorted([int(step * 50) for step in args.steps])
    steps_show = sorted([int(step * 50 + number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: {}\n".format(steps_show))

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, 50)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)
    
    idx_num = 100000
    idx_batch_size = 1000

    # Data reduction
    selected_idx = replay_memory.select_idx(idx_num, idx_batch_size)

    for epoch in range(number_epochs + 1, number_epochs + 51):
        print_and_log(f"{args.experiment_name} epoch {epoch}")
        print(f"Semi-supervised epoch {epoch - number_epochs}")
        
        if (epoch - number_epochs) % 5 == 0:
            print("current student's lr: ", optimizer_S.state_dict()['param_groups'][0]['lr'])

        config.tboard_writer.add_scalar('Param/student_learning_rate', scheduler_S.get_last_lr()[0], args.current_query_count)
        config.tboard_writer.add_scalar('Param/generator_learning_rate', scheduler_G.get_last_lr()[0],
                                 args.current_query_count)

        # Inner training loop call
        # student_preds, teacher_preds = train_epoch_ensemble(args, generator, student, teacher, args.device,
        #                                                     optimizer_S, optimizer_G, epoch, replay_memory)
        student_preds, teacher_preds, s_loss = train_student_ensemble_boost(args, student, generator, args.device, optimizer_S, epoch, replay_memory, selected_idx)

        if args.scheduler != "none":
            scheduler_S.step()
            # scheduler_G.step()
        replay_memory.new_epoch()

        # Test and log
        acc = log_test_metrics(student, test_loader, teacher_test_preds, args.device, args, task="multiclass")
        for goal in accuracy_goals:
            if accuracy_goals[goal] == 0 and acc > goal:
                accuracy_goals[goal] = args.current_query_count / 1000000.0
                print_and_log(f"Reached {goal} accuracy goal in {accuracy_goals[goal]}m queries")

        eval_and_log_validation_metrics(config.tboard_writer, student_preds, teacher_preds, args, task="multiclass")
        log_generator_distribution(config.tboard_writer, teacher_preds, args)

        acc_list.append(acc)
        # Store models
        if acc > best_acc:
            if not os.path.exists(f"{args.experiment_dir}/model_checkpoints"):
                os.makedirs(f"{args.experiment_dir}/model_checkpoints")
            best_acc = acc
            best_acc_epoch = epoch    # add code
            torch.save(student.state_dict(), f"{args.experiment_dir}/model_checkpoints/student_best.pt")
            torch.save(generator.state_dict(), f"{args.experiment_dir}/model_checkpoints/generator_best.pt")
        if epoch % args.generator_store_frequency == 0:
            if not os.path.exists(f"{args.experiment_dir}/model_checkpoints/generators"):
                os.makedirs(f"{args.experiment_dir}/model_checkpoints/generators")
            torch.save(generator.state_dict(),
                       f"{args.experiment_dir}/model_checkpoints/generators/generator_{args.current_query_count}.pt")



    log_hparams(config.tboard_writer, args)
    for goal in sorted(accuracy_goals):
        print_and_log(f"Goal {goal}: {accuracy_goals[goal]}")
    print_and_log("Best Acc=%.6f" % best_acc)
    print_and_log("Best Epoch at %d" % best_acc_epoch)  # add code


    # Visualize Synthetic Data
    z_v = torch.randn((args.batch_size, args.nz)).to(args.device)  # Sample from random number generator
    fake_v = generator(z_v).detach()                            # Generate synthetic data with generator
    for i in range(args.batch_size):
        img = fake_v[i]
        pic = toPIL(img)
        pic.save('./images/img_%d.jpg' % i)


if __name__ == '__main__':
    print("torch version", torch.__version__)
    start = time.time()
    
    main()

    end = time.time()
    seconds = end-start
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("\nthe total time: %d:%02d:%02d" % (h, m, s))
