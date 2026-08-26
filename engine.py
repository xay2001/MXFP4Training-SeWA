# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import json
import math
import os
import sys
from contextlib import nullcontext
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import utils
from pathlib import Path

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils
import dllogger as DLLogger

from quantization import update_BSRamping_period, QLinearLayer

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, eval_data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None,
                    calib_data_loader: Optional[Iterable] = None,
                    scale_optimizer: Optional[torch.optim.Optimizer] = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    
    if args.cosub:
        criterion = torch.nn.BCEWithLogitsLoss()
    
    if "BSRamping" in args.opt:
        if args.opt_ramping_type <= 5:
            CALIBRATE_PERIOD = len(data_loader)
        else:
            raise NotImplementedError
        
        max_step = CALIBRATE_PERIOD
    
    output_dir = Path(args.output_dir)
    scale_update_interval = max(1, getattr(args, 'qlinear_scale_update_interval', 1))
    scale_params = []
    if scale_optimizer is not None:
        for group in scale_optimizer.param_groups:
            scale_params.extend(group['params'])

    def set_scale_mode(optimize, update_history):
        for module in osmq_modules():
            if getattr(module, 'is_scale_sewa', False):
                module.optimize_scale_sewa = optimize
                module.update_scale_history = update_history

    def osmq_modules():
        base_model = model.module if hasattr(model, 'module') else model
        for module in base_model.modules():
            if isinstance(module, QLinearLayer) and getattr(module, 'is_osmq', False):
                yield module

    def osmq_is_ready():
        return any(module._osmq_ready() for module in osmq_modules())

    base_model = model.module if hasattr(model, 'module') else model

    future_utility_monitor_path = getattr(args, 'qlinear_future_monitor_path', '') or \
        os.path.join(args.output_dir, 'future_utility_monitor.jsonl')
    if getattr(args, 'qlinear_future_utility_reset', False) and utils.is_main_process():
        os.makedirs(os.path.dirname(future_utility_monitor_path), exist_ok=True)

    scale_monitor = getattr(args, 'qlinear_scale_monitor', False)
    scale_monitor_interval = max(1, getattr(args, 'qlinear_scale_monitor_interval', 500))
    scale_monitor_path = getattr(args, 'qlinear_scale_monitor_path', '') or \
        os.path.join(args.output_dir, 'scale_monitor.jsonl')
    if getattr(args, 'qlinear_scale_sewa', False):
        scale_monitor_method = 'scale_sewa'
    elif getattr(args, 'qlinear_osmq', False):
        scale_monitor_method = 'osmq'
    else:
        scale_monitor_method = 'qema'
    if scale_monitor and utils.is_main_process():
        os.makedirs(os.path.dirname(scale_monitor_path), exist_ok=True)

    def write_scale_monitor(global_step):
        if not (scale_monitor and utils.is_main_process()):
            return
        layers = {}
        for name, module in base_model.named_modules():
            if isinstance(module, QLinearLayer):
                stats = module.collect_scale_stats()
                if stats is not None:
                    layers[name] = stats
        if not layers:
            return
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "method": scale_monitor_method,
            "layers": layers,
        }
        with open(scale_monitor_path, 'a') as f:
            f.write(json.dumps(record) + "\n")
    
    for step, (samples, targets) in metric_logger.log_every(data_loader, print_freq, header):
        global_step = epoch * len(data_loader) + step
        if "BSRamping" in args.opt and step % CALIBRATE_PERIOD == 0:
            assert max_norm is None, "Only Support no-grad_clipping !!"
            print(f"start calibration of element-wise BSRamping at [epoch#{epoch} step#{step}/{len(data_loader)}]!!!")
            update_BSRamping_period(model, epoch=epoch, step=step, max_step=max_step, calib_data_loader=calib_data_loader,
                                    optimizer=optimizer, loss_scaler=loss_scaler, max_norm=max_norm, device=device, args=args, 
                                    criterion=criterion, mixup_fn=mixup_fn)
        
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        if args.cosub:
            samples = torch.cat((samples,samples),dim=0)
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        set_scale_mode(optimize=False, update_history=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(samples)
            if not args.cosub:
                loss = criterion(samples, outputs, targets)
            else:
                outputs = torch.split(outputs, outputs.shape[0]//2, dim=0)
                loss = 0.25 * criterion(outputs[0], targets) 
                loss = loss + 0.25 * criterion(outputs[1], targets) 
                loss = loss + 0.25 * criterion(outputs[0], outputs[1].detach().sigmoid())
                loss = loss + 0.25 * criterion(outputs[1], outputs[0].detach().sigmoid()) 

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        if scale_optimizer is not None:
            scale_optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        # DFUR consumes the synchronized gradient from this batch as a
        # one-step-delayed counterfactual label, then optionally applies a
        # deterministic sparse reset to the just-updated live master weights.
        future_utility_layers = {}
        if getattr(args, 'qlinear_future_utility_reset', False):
            for name, module in base_model.named_modules():
                if isinstance(module, QLinearLayer) and getattr(module, 'is_future_utility_reset', False):
                    stats = module.update_future_utility_reset(global_step)
                    if stats is not None:
                        future_utility_layers[name] = stats
            if future_utility_layers and utils.is_main_process():
                record = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'method': 'delayed_future_utility_reset',
                    'layers': future_utility_layers,
                }
                with open(future_utility_monitor_path, 'a') as f:
                    f.write(json.dumps(record) + '\n')

        if model_ema is not None:
            model_ema.update(model)
        
        if args.qlinear_ema_decay > 0:         # EMA-Weight Update
            for name, module in base_model.named_modules():
                if isinstance(module, QLinearLayer):
                    if module.training and module.apply_quantize and module.is_ema:
                        module.ema_step += 1
                        module.ema_weight.mul_(module.ema_decay).add_(module.weight.data, alpha = 1 - module.ema_decay)

        scale_loss_value = None
        if scale_optimizer is not None and osmq_is_ready() and (step % scale_update_interval == 0):
            optimizer.zero_grad()
            scale_optimizer.zero_grad()
            set_scale_mode(optimize=True, update_history=False)
            # no_sync() blocks DDP from AllReducing the entire model's grads on the
            # outer pass; we manually sync only scale_params below.
            sync_context = model.no_sync() if hasattr(model, 'no_sync') else nullcontext()
            with sync_context:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    scale_outputs = model(samples)
                    if not args.cosub:
                        scale_loss = criterion(samples, scale_outputs, targets)
                    else:
                        scale_outputs = torch.split(scale_outputs, scale_outputs.shape[0]//2, dim=0)
                        scale_loss = 0.25 * criterion(scale_outputs[0], targets)
                        scale_loss = scale_loss + 0.25 * criterion(scale_outputs[1], targets)
                        scale_loss = scale_loss + 0.25 * criterion(scale_outputs[0], scale_outputs[1].detach().sigmoid())
                        scale_loss = scale_loss + 0.25 * criterion(scale_outputs[1], scale_outputs[0].detach().sigmoid())

                scale_loss_value = scale_loss.item()
                if not math.isfinite(scale_loss_value):
                    print("Scale loss is {}, stopping training".format(scale_loss_value))
                    sys.exit(1)
                scale_loss.backward()

            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                world_size = float(dist.get_world_size())
                handles = []
                for p in scale_params:
                    if p.grad is not None:
                        handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))
                for h in handles:
                    h.wait()
                for p in scale_params:
                    if p.grad is not None:
                        p.grad.div_(world_size)

            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(scale_params, max_norm)

            scale_optimizer.step()
            set_scale_mode(optimize=False, update_history=True)
            optimizer.zero_grad()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if scale_loss_value is not None:
            metric_logger.update(scale_loss=scale_loss_value)

        step_summary = {
            "loss": loss.item(),
        }
        if scale_loss_value is not None:
            step_summary["scale_loss"] = scale_loss_value
        DLLogger.log(step=global_step,
                     data=step_summary, verbosity=0)

        if scale_monitor and (global_step % scale_monitor_interval == 0):
            write_scale_monitor(global_step)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device,
             copied_data_num=0): 
    """ 
    copied_data_num:    ignore results from [-copied_data_num:] 
    
    NOTE: because we copied `copied_data_num` duplicate samples to make 
                len(dataset_val) % batch_size == 0
          in acc calculation, we should ignore them """
    
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    unloaded_num = len(data_loader.dataset)
    for step, (images, target) in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        batch_size = target.shape[0]
        unloaded_num -= batch_size
        if unloaded_num == 0 and copied_data_num > 0:
            target = target[:-copied_data_num]
            batch_size = target.shape[0]
            print(f"eval cut : batch_size = {batch_size}")

        # compute output
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(images)
            
            if unloaded_num == 0 and copied_data_num > 0:
                output = output[:batch_size]
            loss = criterion(output, target)
      
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
