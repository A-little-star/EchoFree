import os
import sys
import time

from pathlib import Path
from collections import defaultdict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.quantization

from logger.logger import get_logger
from loss.loss import mycost, my_crossentropy, sisnr_loss, Mag_Compress_Mse, ccmse_loss

from pyrnnoise.rnnoise_module import MelFilterBank, vorbis_window
from pyrnnoise.main import rnnoise_in, rnnoise_out, rnnoise_getgains

sys.path.append(
    os.path.dirname(__file__))

from torch import autograd
th.autograd.set_detect_anomaly(True)
from loader.datareader_16k_pfdkf import DataReader
import concurrent.futures
import soundfile as sf
from tqdm import tqdm
import copy

def inference(task_id, data_reader, nnet, conf, device, mode, save_dir
              ):
    with th.no_grad():
        print(f'{mode} decoding:')
        for egs in tqdm(data_reader):
            egs = load_obj(egs, device)
            egs["mix"] = egs["mix"].contiguous()
            egs["ref"] = egs["ref"].contiguous()
            egs['laec_out'] = egs['laec_out'].contiguous()
            egs['laec_echo'] = egs['laec_echo'].contiguous()

            inputs = egs['mix'].squeeze(0).detach().cpu()
            laec_out = egs['laec_out'].squeeze(0).detach().cpu()
            far = egs['ref'].squeeze(0).detach().cpu()
            laec_echo = egs['laec_echo'].squeeze(0).detach().cpu()

            inputs = inputs.to(device)
            laec_echo = laec_echo.to(device)

            outputs = nnet(torch.stack([inputs.clone(), laec_echo.clone()], dim=0))
            outputs_wav = outputs["wavs"].squeeze(0)

            # outputs_wav = outputs_wav / torch.max(torch.abs(outputs_wav)) * torch.max(torch.abs(inputs))

            out = outputs_wav.detach().cpu().numpy()

            if not os.path.exists(os.path.join(save_dir, mode)):
                os.makedirs(os.path.join(save_dir, mode))
            save_path = os.path.join(save_dir, mode, egs["utt_id"])
            sf.write(save_path, out, 16000)

def multi_input_forward_hook(input_res, multiply_adds=False):
    input1 = th.randn(1, 133, 114*3)
    input2 = th.randn(1, 3, 128)
    inputs = (input1, input2)
    return inputs

def load_obj(obj, device):
    '''
    Offload tensor object in obj to cuda device
    '''
    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj
    
    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


def reduce_mean(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

class SimpleTimer(object):
    '''
    A simple timer
    '''
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60

class ProgressReporter(object):
    '''
    A sample progress reporter
    '''
    def __init__(self, rank, logger, period=100):
        self.rank = rank
        self.period = period
        if isinstance(logger, str):
            self.logger = get_logger(logger, file=True)
        else:
            self.logger = logger
        self.header = "Trainer"
        self.reset()
    
    def log(self, sstr):
        if self.rank == 0:
            self.logger.info(f"{self.header}: {sstr}")
    
    def eval(self):
        self.log("set eval mode...")
        self.mode = "eval"
        self.reset()
    
    def train(self):
        self.log("set train mode...")
        self.mode = "train"
        self.reset()
    
    def reset(self):
        self.stats = defaultdict(list)
        self.stats_len = 0
        self.timer = SimpleTimer()

    def add(self, key_value_list, batch_num, epoch, max_epoch, lr):
        self.stats_len += 1
        for key, value in key_value_list.items():
            self.stats[key].append(value)

        if not self.stats_len % self.period:
            sstr = ''
            for key, value in key_value_list.items():
                avg = sum(self.stats[key][-self.period:]) / self.period
                sstr += f'| {key} = {avg:+.2f}'
            self.log(f"Epoch {epoch:3d}/{max_epoch:3d} | Batchs {self.stats_len:5d}/{batch_num:5d} {sstr} | lr {lr:.4e}")
        
    def report(self, epoch, lr):
        N = len(self.stats["loss"])
        if self.mode == "eval":
            sstr = ",".join(
                map(lambda f: "{:.2f}".format(f), self.stats["loss"]))
            self.log(f"loss on {N:d} batches: {sstr}")
        
        loss = sum(self.stats["loss"]) / N
        cost = self.timer.elapsed()
        sstr = f"Loss(time/N, lr={lr:.3e}) - Epoch {epoch:2d}: " + f"{self.mode} = {loss:.4f}({cost:.2f}m/{N:d})"
        return loss, sstr

class Trainer(object):
    '''
    Basic neural network trainer
    '''
    def __init__(self,
                 nnet,
                 optimizer,
                 scheduler,
                 device,
                 conf,
                 local_rank,
                 world_size):

        self.default_device = device
        self.local_rank = local_rank
        self.world_size = world_size
        self.conf = conf
        
        self.checkpoint = Path(conf['train']['checkpoint'])
        self.checkpoint.mkdir(exist_ok=True, parents=True)
        self.reporter = ProgressReporter(
            local_rank,
            (self.checkpoint / "trainer.log").as_posix() if conf['logger']['path'] is None else conf['logger']['path'],
            period=conf['logger']['print_freq'])
        
        self.gradient_clip = conf['optim']['gradient_clip']
        self.start_epoch = 0 # zero based
        self.no_impr = conf['train']['early_stop']
        self.save_period = conf['train']['save_period']

        # only network part
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6
        
        # logging
        self.reporter.log("model summary:\n{}".format(nnet))
        self.reporter.log(f"#param: {self.num_params:.2f}M")
        # self.reporter.log(f"#macs: {self.num_macs:.2f}M")
        
        if conf['train']['resume']:
            # resume nnet and optimizer from checkpoint
            if not Path(conf['train']['resume']).exists():
                raise FileNotFoundError(
                    f"Could not find resume checkpoint: {conf['train']['resume']}")
            cpt = th.load(conf['train']['resume'], map_location="cpu")
            self.start_epoch = cpt["epoch"]
            self.reporter.log(
                f"resume from checkpoint {conf['train']['resume']}: epoch {self.start_epoch:d}")
            # load nnet
            # nnet.reload_spk(conf['train']['spk_resume'])

            nnet.load_state_dict(cpt['model_state_dict'], strict=False)
            self.nnet = nnet.to(self.default_device)

            # optimizer.load_state_dict(cpt["optim_state_dict"])
            self.optimizer = optimizer
        else:
            self.nnet = nnet.to(self.default_device)
            self.optimizer = optimizer
        
        if conf['optim']['gradient_clip']:
            self.reporter.log(
                f"gradient clipping by {conf['optim']['gradient_clip']}, default L2")
            self.clip_norm = conf['optim']['gradient_clip']
        else:
            self.clip_norm = 0
        
        self.scheduler = scheduler

        n_fft = 512
        hop_len = 256
        win_len = 512
        fbank_class = MelFilterBank(nfilter=100, nfft=n_fft,
                                sr=16000, lowfreq=0, highfreq=None, transpose=False)
        self.fbank = fbank_class.get_filter_bank()
        self.window = vorbis_window(win_len)
        self.up_scale = 64.0
        
    def save_checkpoint(self, epoch, best=True):
        '''
        Save checkpoint (epoch, model, optimizer)
        '''
        cpt = {
            "epoch": epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        
        cpt_name = "{0}.pt.tar".format("best" if best else "last")
        th.save(cpt, self.checkpoint / cpt_name)
        self.reporter.log(f"save checkpoint {cpt_name}")
        if self.save_period > 0 and epoch % self.save_period == 0:
            th.save(cpt, self.checkpoint / f"{epoch}.pt.tar")
        


    def train(self, data_loader, epoch, max_epoch, lr):
        self.nnet.train()
        self.reporter.train()
        batch_num = len(data_loader) # data_loader是根据参数做好的一个batch的数据，包括训练一个网络需要的所有输入

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.default_device)
            # contiguous跟数据分布有关，建议后面看看
            egs["mix"] = egs["mix"].contiguous()
            egs["far"] = egs["far"].contiguous()
            egs["laec_out"] = egs["laec_out"].contiguous()
            egs["laec_echo"] = egs["laec_echo"].contiguous()
            egs["label"] = egs["label"].contiguous()

            self.optimizer.zero_grad()
            # 跟并行计算有关的

            n_fft = 512
            hop_len = 256
            win_len = 512

            outputs = self.nnet(torch.stack([egs["mix"].clone(), egs["laec_echo"].clone()], dim=0))

            self.fbank = self.fbank.to(self.default_device)
            self.window = self.window.to(self.default_device)
            gains_label = rnnoise_getgains(
                domain="Time",
                inputs=egs["mix"],
                labels=egs["label"],
                Fbank=self.fbank,
                window=self.window,
                n_fft=n_fft,
                hop_length=hop_len,
                win_length=win_len,
                up_scale=self.up_scale
            )
            
            mycost_loss = mycost(gains_label, outputs["gains"])
            my_crossentropy_loss = my_crossentropy(gains_label, outputs["gains"])
            sisnr = sisnr_loss(outputs["wavs"], egs["label"])
            ccmseloss, mag_loss, cplx_loss, asym_loss = ccmse_loss(outputs["wavs"], egs["label"], fft_len=n_fft, hop_len=hop_len, win_len=win_len)
            loss = 10.0 * mycost_loss + 0.5 * my_crossentropy_loss + 0.5 * sisnr + 0.5 * ccmseloss

            loss.backward()

            reduce_cost_loss = reduce_mean(mycost_loss, self.world_size)
            reduce_crossentropy_loss = reduce_mean(my_crossentropy_loss, self.world_size)
            reduce_sisnr_loss = reduce_mean(sisnr, self.world_size)
            reduce_mag_loss = reduce_mean(mag_loss, self.world_size)
            reduce_cplx_loss = reduce_mean(cplx_loss, self.world_size)
            reduce_loss = reduce_mean(loss, self.world_size)

            loss_dict = {}
            loss_dict['COST'] = reduce_cost_loss.item()
            loss_dict['CROSSENTROPY'] = reduce_crossentropy_loss.item()
            loss_dict['SISNR'] = reduce_sisnr_loss.item()
            loss_dict['MAGLOSS'] = reduce_mag_loss.item()
            loss_dict['CPLXLOSS'] = reduce_cplx_loss.item()
            loss_dict['loss'] = reduce_loss.item()

            if self.gradient_clip:
                norm = clip_grad_norm_(self.nnet.parameters(),
                                       self.gradient_clip)
                loss_dict['norm'] = norm.item()
                # self.reporter.add("norm", norm, batch_num, epoch)
            self.reporter.add(loss_dict, batch_num, epoch, max_epoch, lr)
            self.optimizer.step()
    
    def eval(self, data_loader, epoch, max_epoch, lr):
        self.nnet.eval()
        self.reporter.eval()
        batch_num = len(data_loader)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.default_device)
                egs["mix"] = egs["mix"].contiguous()
                egs["far"] = egs["far"].contiguous()
                egs["laec_out"] = egs["laec_out"].contiguous()
                egs["laec_echo"] = egs["laec_echo"].contiguous()
                egs["label"] = egs["label"].contiguous()

                n_fft = 512
                hop_len = 256
                win_len = 512
                self.fbank = self.fbank.to(self.default_device)
                self.window = self.window.to(self.default_device)
                gains_label = rnnoise_getgains(
                    domain="Time",
                    inputs=egs["mix"],
                    labels=egs["label"],
                    Fbank=self.fbank,
                    window=self.window,
                    n_fft=n_fft,
                    hop_length=hop_len,
                    win_length=win_len,
                    up_scale=self.up_scale
                )

                outputs = self.nnet(torch.stack([egs["mix"].clone(), egs["laec_echo"].clone()], dim=0))
                           
                mycost_loss = mycost(gains_label, outputs["gains"])
                my_crossentropy_loss = my_crossentropy(gains_label, outputs["gains"])
                sisnr = sisnr_loss(outputs["wavs"], egs["label"])
                ccmseloss, mag_loss, cplx_loss, asym_loss = ccmse_loss(outputs["wavs"], egs["label"], fft_len=n_fft, hop_len=hop_len, win_len=win_len)
                loss = 10.0 * mycost_loss + 0.5 * my_crossentropy_loss + 0.5 * sisnr + 0.5 * ccmseloss
                
                reduce_cost_loss = reduce_mean(mycost_loss, self.world_size)
                reduce_crossentropy_loss = reduce_mean(my_crossentropy_loss, self.world_size)
                reduce_sisnr_loss = reduce_mean(sisnr, self.world_size)
                reduce_mag_loss = reduce_mean(mag_loss, self.world_size)
                reduce_cplx_loss = reduce_mean(cplx_loss, self.world_size)
                reduce_loss = reduce_mean(loss, self.world_size)

                loss_dict = {}
                loss_dict['COST'] = reduce_cost_loss.item()
                loss_dict['CROSSENTROPY'] = reduce_crossentropy_loss.item()
                loss_dict['SISNR'] = reduce_sisnr_loss.item()
                loss_dict['MAGLOSS'] = reduce_mag_loss.item()
                loss_dict['CPLXLOSS'] = reduce_cplx_loss.item()
                loss_dict['loss'] = reduce_loss.item()
                self.reporter.add(loss_dict, batch_num, epoch, max_epoch, lr)

    def calib(self, data_loader):
        self.nnet.eval()
        with torch.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.default_device)
                egs["mix"] = egs["mix"].contiguous()
                egs["far"] = egs["far"].contiguous()
                egs["laec_out"] = egs["laec_out"].contiguous()
                egs["laec_echo"] = egs["laec_echo"].contiguous()
                egs["label"] = egs["label"].contiguous()

                self.nnet(torch.stack([egs["mix"].clone(), egs["laec_echo"].clone()], dim=0))
    
    def decode(self, save_dir):
        device = torch.device('cpu')
        self.conf["datareader"]["filename"] = self.conf["testlist"]["dt_list"]
        data_reader_dt = DataReader(**self.conf["datareader"])

        self.conf["datareader"]["filename"] = self.conf["testlist"]["fest_list"]
        data_reader_fest = DataReader(**self.conf["datareader"])

        self.conf["datareader"]["filename"] = self.conf["testlist"]["nest_list"]
        data_reader_nest = DataReader(**self.conf["datareader"])

        self.conf["datareader"]["filename"] = self.conf["testlist"]["goer_list"]
        data_reader_goer = DataReader(**self.conf["datareader"])
        if not os.path.exists(self.conf["save"]["dir"] + save_dir):
            os.makedirs(self.conf["save"]["dir"] + save_dir, exist_ok=True) 
        # 参数列表
        model = copy.deepcopy(self.nnet.module).to('cpu')
        quantized_model = torch.quantization.convert(model.eval(), inplace=False)
        # a = torch.randn(2, 1, 160000)
        # print(quantized_model(a))
        params = [
            (0, data_reader_dt, quantized_model, self.conf, device, "dt", self.conf["save"]["dir"] + save_dir),
            (1, data_reader_fest, quantized_model, self.conf, device, "fest", self.conf["save"]["dir"] + save_dir),
            (2, data_reader_nest, quantized_model, self.conf, device, "nest", self.conf["save"]["dir"] + save_dir),
            (3, data_reader_goer, quantized_model, self.conf, device, "goer", self.conf["save"]["dir"] + save_dir)
        ]

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(inference, task_id, data_reader, nnet, conf, device, mode, save_dir) for task_id, data_reader, nnet, conf, device, mode, save_dir in params]
    
            # 等待所有任务完成
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Task generated an exception: {exc}')
    
    def run(self, train_loader, valid_loader, num_epoches=50, test=False):
        '''
        Run on whole training set and evaluate
        '''
        # make dilated conv faster
        th.backends.cudnn.benchmark = True
        # avoid alloc memory grom gpu0
        # th.cuda.set_device(self.default_device)

        if test:
            e = self.start_epoch
            # make sure not inf
            best_loss = 10000
            self.scheduler.best = 10000
            no_impr = 0
        else:
            e = self.start_epoch
            cur_lr = self.optimizer.param_groups[0]["lr"]
            # self.eval(valid_loader, e, num_epoches, cur_lr)
            # best_loss, _ = self.reporter.report(e, 0)
            # self.reporter.log(f"start from epoch {e:d}, loss = {best_loss:.4f}")
            # # make sure not inf
            # self.scheduler.best = best_loss
            no_impr = 0
            best_loss = 50
            self.scheduler.best = best_loss
        
        # self.calib(calib_loader)

        while e < num_epoches:
            e += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]

            if e > 3:
                self.nnet.apply(torch.ao.quantization.disable_observer)
            if e > 2:
                self.nnet.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

            # self.decode('_' + str(e - 1) + 'epoch')
            # >> train
            self.train(train_loader, e, num_epoches, cur_lr)
            _, sstr = self.reporter.report(e, cur_lr)
            self.reporter.log(sstr)
            # << train
            # >> eval
            self.eval(valid_loader, e, num_epoches, cur_lr)
            cv_loss, sstr = self.reporter.report(e, cur_lr)
            if cv_loss > best_loss:
                no_impr += 1
                sstr += f"| no impr, best = {self.scheduler.best:.4f}"
            else:
                best_loss = cv_loss
                no_impr = 0
                self.save_checkpoint(e, best=True)
            self.reporter.log(sstr)
            # << eval
            # schedule here
            self.scheduler.step(cv_loss)
            # flush scheduler info
            sys.stdout.flush()
            # save checkpoint
            self.save_checkpoint(e, best=False)
            if no_impr == self.no_impr:
                self.reporter.log(
                    f"stop training cause no impr for {no_impr:d} epochs")
                break
        self.reporter.log(f"training for {e:d}/{num_epoches:d} epoches done!")
