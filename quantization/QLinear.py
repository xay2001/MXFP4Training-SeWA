import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function

from .Qconfig import qconfig
from .utils import *
from .QFunction import *

import os
from copy import deepcopy

from torch.cuda import amp

class QLinearLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, args=None, layer_type=''):
        super(QLinearLayer, self).__init__(in_features, out_features, bias)
        self.args = args
        self.layer_type = layer_type
        assert layer_type != '', "layer_type is not defined"
        assert layer_type in qconfig.qlinear_config.keys(), f"{layer_type} not in qlinear_config"

        self.apply_quantize = list_has_common_element(args.qchoice, qconfig.qlinear_config[layer_type])

        self.fbit = self.args.fwbit if self.args.fwbit else self.Ubit
        self.bbit = self.args.bwbit if self.args.bwbit else self.Ubit
        quantize_flag = format_string_with_condition(layer_type, self.apply_quantize, self.args.symm, self.fbit, self.bbit,
                                                     self.args.row_blocksize, self.args.column_blocksize)
        print(quantize_flag)
        self.frozen_args = deepcopy(self.args)
        
        if self.apply_quantize and args.qlinear_ema_decay > 1e-10:
            self.is_ema = True
            self.ema_decay = args.qlinear_ema_decay
            
            self.register_buffer('ema_step', torch.tensor(0, dtype=torch.int))
            self.register_buffer('ema_weight', self.weight.data.clone())
        else:
            self.is_ema = False

        self.is_osmq = self.apply_quantize and getattr(args, 'qlinear_osmq', False)
        if self.is_osmq:
            Bweight = block_cut(self.weight.data, args.row_blocksize, args.column_blocksize)
            self.mask_logit = nn.Parameter(torch.zeros(Bweight.shape[0], 3))
            self.mask_logit.data[:, 0] = getattr(args, 'qlinear_osmq_init_current_bias', 1.0)
            self.register_buffer('prev_rqweight', self.weight.data.clone())
            self.register_buffer('prev_osmq_mask', torch.zeros(Bweight.shape[0], 3))
            self.prev_osmq_mask[:, 0] = 1.0
            self.is_scale_sewa = getattr(args, 'qlinear_scale_sewa', False)
            if self.is_scale_sewa:
                self.scale_mask_logit = nn.Parameter(torch.zeros(Bweight.shape[0], 4))
                self.scale_mask_logit.data[:, 0] = getattr(args, 'qlinear_scale_sewa_init_current_bias', 1.0)
                self.register_buffer('wscale_history', torch.ones(Bweight.shape[0], 4, 1, 1))
                self.register_buffer('scale_history_step', torch.tensor(0, dtype=torch.long))
        else:
            self.is_scale_sewa = False

    def _osmq_ready(self):
        if not self.is_osmq or not self.is_ema:
            return False
        start_step = getattr(self.args, 'qlinear_osmq_start_step', 0)
        return self.ema_step >= max(10, start_step)

    def _compute_gumbel_mask(self, logit, tau):
        tau = max(tau, 1e-6)

        if self.training:
            eps = 1e-9
            u = torch.rand_like(logit)
            gumbel = -torch.log(-torch.log(u + eps) + eps)
            y = F.softmax((logit + gumbel) / tau, dim=-1)
        else:
            y = F.softmax(logit / tau, dim=-1)

        index = y.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(y).scatter_(-1, index, 1.0)
        return (y_hard - y).detach() + y

    def _compute_osmq_mask(self):
        return self._compute_gumbel_mask(
            self.mask_logit,
            getattr(self.args, 'qlinear_osmq_tau', 1.0)
        )

    def _compute_scale_sewa_weight(self):
        tau = getattr(self.args, 'qlinear_scale_sewa_tau', 1.0)
        tau = max(tau, 1e-6)
        if self.training:
            eps = 1e-9
            u = torch.rand_like(self.scale_mask_logit)
            gumbel = -torch.log(-torch.log(u + eps) + eps)
            weight = F.softmax((self.scale_mask_logit + gumbel) / tau, dim=-1)
            if not getattr(self, 'optimize_scale_sewa', False):
                weight = weight.detach()
            return weight
        return F.softmax(self.scale_mask_logit / tau, dim=-1)

    def _fp_quant_params(self, bits, exp_bits):
        m_bit = bits - 1 - exp_bits
        elow = -2 ** (exp_bits - 1) + 2
        ehigh = elow + 2 ** exp_bits - 2
        mhigh = 2 ** m_bit - 1
        qmax = (1 + mhigh / (mhigh + 1)) * (2 ** ehigh)
        qmin = -qmax
        return elow, ehigh, mhigh, qmin, qmax

    def _compute_fp_scale(self, input):
        absmax_per_block = input.abs().amax(dim=(1, 2)).unsqueeze(1).unsqueeze(2)
        absmax_per_block = torch.where(
            absmax_per_block == 0,
            absmax_per_block + self.args.epsilon,
            absmax_per_block)

        if self.args.fwbit in (100, 32, 16) or not (self.apply_quantize and (self.args.qlinear_f_w_in or self.args.qlinear_all)):
            return torch.ones_like(absmax_per_block)

        _, ehigh, _, qmin, qmax = self._fp_quant_params(self.args.fwbit, self.args.fwexp)
        scale_per_block = (2 * absmax_per_block) / (qmax - qmin)
        scale_per_block = scale_per_block.to(input)

        if self.args.mxscale == 1:
            scale_per_block = torch.ceil(torch.log2(scale_per_block))
            scale_per_block = torch.clamp(scale_per_block, -127, 127)
            scale_per_block = 2 ** scale_per_block
        elif self.args.mxscale == 2:
            scale_per_block = torch.floor(torch.log2(scale_per_block))
            scale_per_block = torch.clamp(scale_per_block, -127, 127)
            scale_per_block = 2 ** scale_per_block
        elif self.args.mxscale == 3:
            scale_per_block = torch.floor(torch.log2(absmax_per_block)) - ehigh
            scale_per_block = torch.clamp(scale_per_block, -127, 127)
            scale_per_block = 2 ** scale_per_block
        return scale_per_block

    def _cast_to_fp_with_scale(self, input, scale, ema_input=None):
        if self.args.fwbit == 100 or not (self.apply_quantize and (self.args.qlinear_f_w_in or self.args.qlinear_all)):
            return input, input
        if self.args.fwbit == 32:
            return input.to(torch.float32), input.to(torch.float32)
        if self.args.fwbit == 16:
            return input.to(torch.bfloat16), input.to(torch.bfloat16)

        elow, _, mhigh, qmin, qmax = self._fp_quant_params(self.args.fwbit, self.args.fwexp)
        x = input / scale
        sign = x.sign()
        x_abs = x.abs()
        expo = torch.floor(torch.log2(x_abs + self.args.epsilon))
        expo = torch.clamp(expo, min=elow)

        if ema_input is None:
            mant = x_abs / (2 ** expo)
            mant_int = torch.floor(mant)
            mant_frac = (mant - mant_int) * (mhigh + 1)
            mant_frac = torch.round(torch.clamp(mant_frac, 0, mhigh + 1))
            mant_q = mant_int + mant_frac / (mhigh + 1)
            qinput = sign * (2 ** expo) * mant_q
        else:
            expo_value = 2 ** expo
            ema_x = ema_input / scale
            ema_x_signed = sign * ema_x
            mant = x_abs / expo_value
            mant_ema = ema_x_signed / expo_value
            mant_int = torch.floor(mant)
            mant_frac = (mant - mant_int) * (mhigh + 1)
            mant_ema_frac = (mant_ema - mant_int) * (mhigh + 1)
            mant_frac = torch.floor(torch.clamp(mant_frac, 0, mhigh + 1))
            mant_frac = mant_frac + (mant_ema_frac >= mant_frac + 0.5).to(mant_frac)
            mant_q = mant_int + mant_frac / (mhigh + 1)
            qinput = sign * expo_value * mant_q

        qinput = torch.clamp(qinput, qmin, qmax).to(input)
        return qinput * scale, qinput

    def _update_wscale_history(self, scale=None):
        if not (self.is_scale_sewa and self.training and self._osmq_ready()
                and getattr(self, 'update_scale_history', True)):
            return

        with torch.no_grad():
            if scale is None:
                Bweight = block_cut(self.weight, self.args.row_blocksize, self.args.column_blocksize)
                scale = self._compute_fp_scale(Bweight)
            scale = scale.detach()
            interval = max(1, getattr(self.args, 'qlinear_scale_update_interval', 1))
            step = int(self.scale_history_step.item())
            if step == 0:
                self.wscale_history.copy_(scale.unsqueeze(1).expand_as(self.wscale_history))
            elif step % interval == 0:
                self.wscale_history[:, 3].copy_(self.wscale_history[:, 2])
                self.wscale_history[:, 2].copy_(self.wscale_history[:, 1])
                self.wscale_history[:, 1].copy_(self.wscale_history[:, 0])
                self.wscale_history[:, 0].copy_(scale)
            self.scale_history_step.add_(1)

    def _standard_rqweight(self, ema_weight=None):
        with torch.no_grad():
            Bweight = block_cut(self.weight, self.args.row_blocksize, self.args.column_blocksize)
            Bema_weight = None
            if ema_weight is not None:
                Bema_weight = block_cut(ema_weight, self.args.row_blocksize, self.args.column_blocksize)

            RQweight, _, _ = block_quant(
                Bweight, self.args.symm, self.args.fwbit, exp_bits=self.args.fwexp,
                stochastic=False, epsilon=self.args.epsilon,
                apply_quantize=self.apply_quantize and (self.args.qlinear_f_w_in or self.args.qlinear_all),
                ema_input=Bema_weight, use_triton=self.args.tritonQ,
                MXScale=self.args.mxscale)
            return block_reshape(RQweight, self.weight, self.args.row_blocksize, self.args.column_blocksize)

    def _update_prev_rqweight(self, rqweight):
        if self.is_osmq and self.training:
            with torch.no_grad():
                self.prev_rqweight.copy_(rqweight.detach())

    def _osmq_rqweight(self):
        if self.is_scale_sewa:
            return self._osmq_rqweight_scale_sewa()

        Bweight = block_cut(self.weight, self.args.row_blocksize, self.args.column_blocksize)
        Bema_weight = block_cut(self.ema_weight, self.args.row_blocksize, self.args.column_blocksize)
        Bprev_weight = block_cut(self.prev_rqweight, self.args.row_blocksize, self.args.column_blocksize)

        RQcurrent, _, _ = block_quant(
            Bweight, self.args.symm, self.args.fwbit, exp_bits=self.args.fwexp,
            stochastic=False, epsilon=self.args.epsilon,
            apply_quantize=self.apply_quantize and (self.args.qlinear_f_w_in or self.args.qlinear_all),
            ema_input=None, use_triton=self.args.tritonQ,
            MXScale=self.args.mxscale)
        RQema, _, _ = block_quant(
            Bweight, self.args.symm, self.args.fwbit, exp_bits=self.args.fwexp,
            stochastic=False, epsilon=self.args.epsilon,
            apply_quantize=self.apply_quantize and (self.args.qlinear_f_w_in or self.args.qlinear_all),
            ema_input=Bema_weight, use_triton=self.args.tritonQ,
            MXScale=self.args.mxscale)

        candidates = torch.stack((RQcurrent.detach(), RQema.detach(), Bprev_weight.detach()), dim=1)
        mask = self._compute_osmq_mask().to(candidates)
        selected = (mask.view(mask.shape[0], 3, 1, 1) * candidates).sum(dim=1)
        rqweight = block_reshape(selected, self.weight, self.args.row_blocksize, self.args.column_blocksize)
        rqweight = rqweight + (self.weight - self.weight.detach())

        if self.training:
            with torch.no_grad():
                self.prev_rqweight.copy_(rqweight.detach())
                self.prev_osmq_mask.copy_(mask.detach())
        return rqweight

    def _osmq_rqweight_scale_sewa(self):
        Bweight = block_cut(self.weight, self.args.row_blocksize, self.args.column_blocksize)
        Bema_weight = block_cut(self.ema_weight, self.args.row_blocksize, self.args.column_blocksize)
        Bprev_weight = block_cut(self.prev_rqweight, self.args.row_blocksize, self.args.column_blocksize)

        scale_current = self._compute_fp_scale(Bweight).detach()
        if self.training and int(self.scale_history_step.item()) == 0:
            with torch.no_grad():
                self.wscale_history.copy_(scale_current.unsqueeze(1).expand_as(self.wscale_history))
        scale_history = self.wscale_history.to(Bweight).detach()
        scale_candidates = torch.stack(
            (scale_current, scale_history[:, 0], scale_history[:, 1], scale_history[:, 3]),
            dim=1)
        scale_weight = self._compute_scale_sewa_weight().to(scale_candidates)
        selected_scale = (scale_weight.view(scale_weight.shape[0], 4, 1, 1) * scale_candidates).sum(dim=1)

        RQcurrent, _ = self._cast_to_fp_with_scale(Bweight, selected_scale)
        RQema, _ = self._cast_to_fp_with_scale(Bweight, selected_scale, ema_input=Bema_weight)

        candidates = torch.stack((RQcurrent, RQema, Bprev_weight.detach()), dim=1)
        mask = self._compute_osmq_mask().to(candidates)
        selected = (mask.view(mask.shape[0], 3, 1, 1) * candidates).sum(dim=1)
        rqweight = block_reshape(selected, self.weight, self.args.row_blocksize, self.args.column_blocksize)
        rqweight = rqweight + (self.weight - self.weight.detach())

        # Outer (scale-only) pass must NOT mutate inner-pass state.
        if self.training and not getattr(self, 'optimize_scale_sewa', False):
            with torch.no_grad():
                self.prev_rqweight.copy_(rqweight.detach())
                self.prev_osmq_mask.copy_(mask.detach())
                self._update_wscale_history(scale_current)
        return rqweight

    def forward(self, input):
        
        """
        We update W_EMA During Training Process (in engine.py/train_one_epoch)
        if args.qlinear_ema_decay > 0:         # EMA-Weight Update
            for name, module in model.module.named_modules():
                if isinstance(module, QLinearLayer):
                    if module.training and module.apply_quantize and module.is_ema:
                        module.ema_step += 1
                        module.ema_weight.mul_(module.ema_decay) \
                                         .add_(module.weight.data, alpha = 1 - module.ema_decay)
        """
        if self.apply_quantize and self._osmq_ready():
            rqweight = self._osmq_rqweight()
            output = QuantLinearWithRQWeight.apply(input,
                                    rqweight, self.bias,
                                    self.args, self.layer_name, self.apply_quantize)
        elif self.apply_quantize and self.is_ema and self.ema_step >= 10 and self.args.qlinear_ema_decay > 1e-10:
            output = QuantLinear.apply(input, 
                                    self.weight, self.bias, self.ema_weight,
                                    self.args, self.layer_name, self.apply_quantize)
            if self.is_osmq:
                self._update_prev_rqweight(self._standard_rqweight(self.ema_weight))
        else:
            output = QuantLinear.apply(input, 
                                    self.weight, self.bias, None,
                                    self.args, self.layer_name, self.apply_quantize)
            if self.is_osmq:
                self._update_prev_rqweight(self._standard_rqweight(None))

        if self.is_osmq and not self._osmq_ready():
            output = output + self.mask_logit.sum() * 0.0
        if self.is_scale_sewa and self.training and not getattr(self, 'optimize_scale_sewa', False):
            output = output + self.scale_mask_logit.sum() * 0.0

        return output

class QuantLinearWithRQWeight(Function):
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, input, rqweight, bias,
                args, layer_name, apply_quantize=True):

        ######## [Q1] Qinput, Iscale = Q_D(X)  1x32 ##########
        Binput = block_cut(input, args.row_blocksize, args.column_blocksize)
        RQinput, Qinput, Iscale = block_quant(
            Binput, args.symm, args.fabit, exp_bits=args.faexp,
            stochastic=False, epsilon=args.epsilon,
            apply_quantize=apply_quantize and (args.qlinear_f_a_in or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)

        Qinput  = block_reshape( Qinput, input, args.row_blocksize, args.column_blocksize)
        RQinput = block_reshape(RQinput, input, args.row_blocksize, args.column_blocksize)

        ctx.saved = Qinput, Iscale, rqweight, bias, args, layer_name, apply_quantize
        fc_output = F.linear(RQinput, rqweight, bias)

        return fc_output

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        Qinput, Iscale, rqweight, bias, args, layer_name, apply_quantize = ctx.saved

        Bgrad_output   = block_cut(grad_output, args.row_blocksize,    args.column_blocksize)
        Bgrad_output_t = block_cut(grad_output, args.column_blocksize, args.row_blocksize)

        ######## [Q3] Q_S ( ∇_Y )  1x32 ##########
        RQgrad_output, Qgrad_output, _ = block_quant(
            Bgrad_output, args.symm, args.babit, exp_bits=args.baexp,
            stochastic=True, epsilon=args.epsilon,
            apply_quantize=apply_quantize and (args.qlinear_b_dy or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)

        ######## [Q5] Q_S ( ∇_Y )  32x1 ##########
        RQgrad_output_t, Qgrad_output_t, _ = block_quant(
            Bgrad_output_t, args.symm, args.babit, exp_bits=args.bwexp,
            stochastic=True, epsilon=args.epsilon,
            apply_quantize=apply_quantize and (args.qlinear_b_dy_t or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)

        grad_output   = block_reshape(RQgrad_output,   grad_output, args.row_blocksize,    args.column_blocksize)
        grad_output_t = block_reshape(RQgrad_output_t, grad_output, args.column_blocksize, args.row_blocksize)

        #=======================∇_W Calculation=====================
        Binput = block_cut(Qinput, args.row_blocksize, args.column_blocksize)
        Dinput = Binput * Iscale
        Dinput = block_reshape(Dinput, Qinput, args.row_blocksize, args.column_blocksize)

        BRinput = block_cut(Dinput, args.column_blocksize, args.row_blocksize)
        QRinput, _, _ = block_quant(
            BRinput, args.symm, args.fabit, exp_bits=args.faexp,
            stochastic=True, epsilon=args.epsilon,
            apply_quantize=apply_quantize and (args.qlinear_b_a_rq or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)

        Rinput = block_reshape(QRinput, Qinput, args.column_blocksize, args.row_blocksize)

        grad_output_flatten_t = grad_output_t.reshape(-1, grad_output_t.shape[-1])
        input_flatten         = Rinput.reshape(-1, Qinput.shape[-1])
        grad_rqweight = grad_output_flatten_t.t().mm(input_flatten)

        #=======================∇_X Calculation=====================
        BRweight = block_cut(rqweight, args.column_blocksize, args.row_blocksize)
        QRweight, _, _ = block_quant(
            BRweight, args.symm, args.fwbit, exp_bits=args.fwexp,
            stochastic=True, epsilon=args.epsilon,
            apply_quantize=apply_quantize and (args.qlinear_b_w_rq or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)

        Rweight = block_reshape(QRweight, rqweight, args.column_blocksize, args.row_blocksize)

        grad_output_flatten = grad_output.reshape(-1, grad_output_t.shape[-1])
        grad_input = grad_output_flatten.mm(Rweight)
        grad_input_transform = grad_input.reshape(Qinput.size())

        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None

        return grad_input_transform, grad_rqweight, grad_bias, None, None, None

class QuantLinear(Function):
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, input, weight, bias, ema_weight,
                args, layer_name, apply_quantize=True):
        
        ######## [Q1] Qinput, Iscale = Q_D(X)  1x32 ##########
        Binput = block_cut(input, args.row_blocksize, args.column_blocksize)
        RQinput, Qinput, Iscale = block_quant(
            Binput, args.symm, args.fabit, exp_bits=args.faexp,
            stochastic=False, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_f_a_in or args.qlinear_all), 
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        Qinput  = block_reshape( Qinput, input, args.row_blocksize, args.column_blocksize)
        RQinput = block_reshape(RQinput, input, args.row_blocksize, args.column_blocksize)

        ######## [Q2] Qweight, Wscale = Q_D(W)  1x32 ##########
        ### NOTE: We leave transpose to F.linear, so the blocksize here is 1x32
        
        Bweight = block_cut(weight, args.row_blocksize, args.column_blocksize)
        
        if not (ema_weight is None):
            Bema_weight = block_cut(ema_weight, args.row_blocksize, args.column_blocksize)
        else:
            Bema_weight = None
        
        RQweight, Qweight, Wscale = block_quant(
            Bweight, args.symm, args.fwbit, exp_bits=args.fwexp,
            stochastic=False, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_f_w_in or args.qlinear_all),
            ema_input=Bema_weight, use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        Qweight = block_reshape(Qweight, weight, args.row_blocksize, args.column_blocksize)
        RQweight = block_reshape(RQweight, weight, args.row_blocksize, args.column_blocksize)

        # save Q_D(X), Q_D(W) for backward
        ctx.saved = Qinput, Iscale, Qweight, Wscale, bias, args, layer_name, apply_quantize
        
        # Linear output
        fc_output = F.linear(RQinput, RQweight, bias) 
        
        return fc_output

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        Qinput, Iscale, Qweight, Wscale, bias, args, layer_name, apply_quantize = ctx.saved

        Bgrad_output   = block_cut(grad_output, args.row_blocksize,    args.column_blocksize)  # RB x CB = 1 x 32
        Bgrad_output_t = block_cut(grad_output, args.column_blocksize, args.row_blocksize)     # CB x RB = 32 x 1
        
        ######## [Q3] Q_S ( ∇_Y )  1x32 ##########
        RQgrad_output, Qgrad_output, _ = block_quant(
            Bgrad_output, args.symm, args.babit, exp_bits=args.baexp,
            stochastic=True, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_b_dy or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        ######## [Q5] Q_S ( ∇_Y )  32x1 ##########
        ### NOTE: We leave transpose outside quantization, so the blocksize here is 1x32
        RQgrad_output_t, Qgrad_output_t, _ = block_quant(
            Bgrad_output_t, args.symm, args.babit, exp_bits=args.bwexp,
            stochastic=True, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_b_dy_t or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        # get quantize-dequantize   ∇_Y(1x32) & ∇_Y(32x1)
        grad_output   = block_reshape(RQgrad_output,   grad_output, args.row_blocksize,    args.column_blocksize)
        grad_output_t = block_reshape(RQgrad_output_t, grad_output, args.column_blocksize, args.row_blocksize)

        #=======================∇_W Calculation=====================
        ######## [Q6]  Q_S ( Dequantize[Q_D(X)] 1x32 )  32x1 ########
        # [Q6]-[1]      Dequantize[Q_D(X)] 1x32
        Binput = block_cut(Qinput, args.row_blocksize, args.column_blocksize)
        Dinput = Binput * Iscale
        Dinput = block_reshape(Dinput, Qinput, args.row_blocksize, args.column_blocksize)
        
        # [Q6]-[2]      Q_S ( Dequantize[Q_D(X)] 1x32 )  32x1
        BRinput = block_cut(Dinput, args.column_blocksize, args.row_blocksize)            
        QRinput, _, _ = block_quant( # [x] TODO: determine `dtype` here 
            BRinput, args.symm, args.fabit, exp_bits=args.faexp, 
            stochastic=True, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_b_a_rq or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        Rinput = block_reshape(QRinput, Qinput, args.column_blocksize, args.row_blocksize)

        grad_output_flatten_t = grad_output_t   .reshape(-1, grad_output_t.shape[-1])
        input_flatten         = Rinput          .reshape(-1, Qinput.shape[-1])

        #   ∇_W Calculation
        grad_weight = grad_output_flatten_t.t().mm(input_flatten)
        
        #=======================∇_X Calculation=====================
        ######## [Q4] Q_S ( Dequantize[Q_D(W)] 1x32 )  32x1 ########
        # [Q4]-[1]      Dequantize[Q_D(W)] 1x32
        Bweight = block_cut(Qweight, args.row_blocksize, args.column_blocksize)
        Dweight = Bweight * Wscale
        Dweight = block_reshape(Dweight, Qweight, args.row_blocksize, args.column_blocksize) 
        
        # [Q4]-[2]      Q_S ( Dequantize[Q_D(W)] 1x32 )  32x1
        BRweight = block_cut(Dweight, args.column_blocksize, args.row_blocksize)            
        QRweight, _, _ = block_quant( # [x] TODO: determine `dtype` here 
            BRweight, args.symm, args.fwbit, exp_bits=args.fwexp, 
            stochastic=True, epsilon=args.epsilon, 
            apply_quantize=apply_quantize and (args.qlinear_b_w_rq or args.qlinear_all),
            use_triton=args.tritonQ,
            MXScale=args.mxscale)
        
        Rweight = block_reshape(QRweight, Qweight, args.column_blocksize, args.row_blocksize)

        # ∇_X Calculation
        grad_output_flatten = grad_output.reshape(-1, grad_output_t.shape[-1])
        grad_input = grad_output_flatten.mm(Rweight)
        grad_input_transform = grad_input.reshape(Qinput.size())
        
        # ∇_Bias Calculation
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None

        return grad_input_transform, grad_weight, grad_bias, None, None, None, None