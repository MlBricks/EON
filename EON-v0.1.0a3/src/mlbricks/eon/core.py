from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

ComponentSpec = str | nn.Module | Callable[..., nn.Module]


class ConfigurationError(ValueError):
    pass


def _expand(value, depth: int, name: str):
    if isinstance(value, (list, tuple)):
        if len(value) != depth:
            raise ConfigurationError(f"{name} must contain exactly depth={depth} entries")
        return list(value)
    return [value for _ in range(depth)]


def _expand_config(value, depth: int, name: str):
    if value is None:
        return [{} for _ in range(depth)]
    if isinstance(value, Mapping):
        return [dict(value) for _ in range(depth)]
    if isinstance(value, (list, tuple)):
        if len(value) != depth:
            raise ConfigurationError(f"{name} must contain exactly depth={depth} entries")
        return [dict(v or {}) for v in value]
    raise TypeError(f"{name} must be a mapping, sequence of mappings, or None")


def _instantiate_custom(spec, config: dict[str, Any], *, kind: str) -> nn.Module:
    if isinstance(spec, nn.Module):
        if config:
            raise ConfigurationError(f"{kind}_config cannot be used with an already-created nn.Module")
        return spec
    if not callable(spec):
        raise TypeError(f"custom {kind} must be an nn.Module or callable factory")
    module = spec(**config)
    if not isinstance(module, nn.Module):
        raise TypeError(f"custom {kind} factory must return nn.Module")
    return module


def _call_with_supported_kwargs(cls, kwargs: dict[str, Any]):
    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls(**kwargs)
    accepts_var = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_var:
        return cls(**kwargs)
    allowed = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**allowed)


def _load_mixer_class(name: str):
    key = name.lower().strip()
    if key == "esa":
        from mlbricks import ESA
        return ESA
    if key == "bolt":
        try:
            from mlbricks import BOLT
            return BOLT
        except ImportError:
            from mlbricks import Bolt
            return Bolt
    raise ConfigurationError(f"unknown mixer {name!r}; expected 'esa', 'bolt', or a custom module/factory")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.float()
        z = z * torch.rsqrt(z.square().mean(-1, keepdim=True) + self.eps)
        return (z * self.weight.float()).to(x.dtype)


class ConventionalFFN(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, activation: str = "silu"):
        super().__init__()
        hidden = int(hidden or 4 * dim)
        self.in_proj = nn.Linear(dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, dim, bias=False)
        self.activation = activation.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        if self.activation == "silu": h = F.silu(h)
        elif self.activation == "gelu": h = F.gelu(h)
        elif self.activation == "relu": h = F.relu(h)
        else: raise ConfigurationError(f"unsupported activation {self.activation!r}")
        return self.out_proj(h)


class PlainFFNAdapter(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__(); self.module = module
    def forward(self, x, current_context, previous_context, state):
        return self.module(x), state


class StateAwareFFNAdapter(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__(); self.module = module
    def forward(self, x, current_context, previous_context, state):
        return self.module(x, current_context, previous_context, state)


def _is_state_aware(module: nn.Module) -> bool:
    try: sig = inspect.signature(module.forward)
    except (TypeError, ValueError): return True
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params): return True
    positional = [p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    return len(positional) >= 4


@dataclass
class _LayerFastPlan:
    layer: nn.Module
    mixer_backend: nn.Module
    input_weight: torch.Tensor
    input_bias: torch.Tensor
    depth_terms: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    candidate_scale: torch.Tensor
    write_scale: torch.Tensor
    delta_mag_scale: torch.Tensor
    mix_scale: torch.Tensor
    dim: int
    state_dim: int


def _unwrap_esa_backend(mixer: nn.Module) -> nn.Module:
    # MLBricks ESA is a public wrapper around ThunderESA in ``mixer.layer``.
    # torch.compile may add one or more ``_orig_mod`` wrappers. Generation
    # packing must operate on the actual trained qgv/out_proj tensors.
    target = getattr(mixer, "layer", mixer)
    seen = set()
    while hasattr(target, "_orig_mod") and id(target) not in seen:
        seen.add(id(target)); target = target._orig_mod
    return target


def _can_fastpath_layer(layer: nn.Module) -> bool:
    mixer = _unwrap_esa_backend(layer.mixer)
    ffn = layer.ffn
    return (
        hasattr(mixer, "qgv") and hasattr(mixer, "out_proj")
        and hasattr(mixer, "head") and hasattr(mixer, "head_dim")
        and hasattr(mixer, "gate_min") and hasattr(mixer, "gate_max") and hasattr(mixer, "eps")
        and hasattr(ffn, "x_proj") and hasattr(ffn, "context_candidate")
        and hasattr(ffn, "context_write") and hasattr(ffn, "state_proj")
        and hasattr(ffn, "output") and hasattr(ffn, "depth_proj")
        and hasattr(ffn, "depth_embedding") and hasattr(ffn, "state_dim")
    )


def _build_layer_fast_plan(layer: nn.Module) -> _LayerFastPlan:
    if not _can_fastpath_layer(layer):
        raise ConfigurationError("generation fast path requires built-in ESA + SAFFN-compatible layer")
    mixer_backend = _unwrap_esa_backend(layer.mixer)
    d = int(layer.norm.weight.numel())
    s = int(layer.ffn.state_dim)
    w = torch.cat([mixer_backend.qgv.weight.detach(), layer.ffn.x_proj.weight.detach()], dim=0).contiguous()
    b = torch.cat([
        torch.zeros(3*d, device=w.device, dtype=layer.ffn.x_proj.bias.dtype),
        layer.ffn.x_proj.bias.detach(),
    ], dim=0).contiguous()
    depth = layer.ffn.depth_proj(layer.ffn.depth_embedding).detach()
    dc, dw, dv = depth.split(s, dim=-1)
    return _LayerFastPlan(
        layer=layer, mixer_backend=mixer_backend, input_weight=w, input_bias=b, depth_terms=(dc, dw, dv),
        candidate_scale=torch.sigmoid(layer.ffn.candidate_transition_logit).detach(),
        write_scale=torch.sigmoid(layer.ffn.write_transition_logit).detach(),
        delta_mag_scale=torch.exp(layer.ffn.delta_magnitude_log_scale).detach(),
        mix_scale=torch.sigmoid(layer.mix).detach(), dim=d, state_dim=s,
    )


def _packed_esa_and_xproj(z: torch.Tensor, mixer_state: torch.Tensor, plan: _LayerFastPlan):
    d, s, layer = plan.dim, plan.state_dim, plan.layer
    mixer = plan.mixer_backend
    packed = F.linear(z, plan.input_weight, plan.input_bias)
    qgv, xall = packed[..., :3*d], packed[..., 3*d:]
    q, gate_raw, value_raw = qgv.split(d, dim=-1)
    xc, xw, xv = xall.split(s, dim=-1)
    B, T, _ = z.shape
    H, HD = int(mixer.head), int(mixer.head_dim)
    q = q.reshape(B, T, H, HD)
    gate_raw = gate_raw.reshape(B, T, H, HD)
    value_raw = value_raw.reshape(B, T, H, HD)
    gate = torch.sigmoid(gate_raw)
    A = mixer.gate_min + (mixer.gate_max - mixer.gate_min) * gate
    V = torch.tanh(value_raw)
    B_write = (1.0 - A) * V
    new_state = A[:, 0].to(mixer_state.dtype) * mixer_state + B_write[:, 0].to(mixer_state.dtype)
    E = new_state.unsqueeze(1).reshape(B, T, d)
    qflat = q.reshape(B, T, d).to(E.dtype)
    E = E * torch.rsqrt(E.pow(2).mean(dim=-1, keepdim=True) + mixer.eps)
    context = mixer.out_proj((torch.sigmoid(qflat) * E).to(z.dtype))
    if hasattr(mixer, "dropout"):
        context = mixer.dropout(context)
    return context, new_state.contiguous(), xc, xw, xv


def _fast_saffn(context, previous_context, state, xc, xw, xv, plan: _LayerFastPlan, *, first_layer: bool):
    ffn, s = plan.layer.ffn, plan.state_dim
    dc, dw, dv = plan.depth_terms
    if first_layer:
        delta = context
        candidate_context = context + plan.candidate_scale * context
        write_context = context + plan.write_scale * context
        candidate = torch.tanh(xc + ffn.context_candidate(candidate_context) + dc)
        write = torch.sigmoid(xw + ffn.context_write(write_context) + dw)
        next_state = write * candidate
    else:
        delta = context - previous_context
        candidate_context = context + plan.candidate_scale * delta
        write_context = context + plan.write_scale * delta
        sc, sw = ffn.state_proj(state).split(s, dim=-1)
        candidate = torch.tanh(xc + ffn.context_candidate(candidate_context) + sc + dc)
        write = torch.sigmoid(xw + ffn.context_write(write_context) + sw + dw)
        dm = torch.sqrt(delta.float().square().mean(-1, keepdim=True) + 1e-6).to(context.dtype)
        scaled = plan.delta_mag_scale * dm
        retain = torch.sigmoid(ffn.retain_logit - scaled * ffn.retain_delta_scale)
        next_state = (1.0 - write) * (retain * state) + write * candidate
    if first_layer:
        dm = torch.sqrt(delta.float().square().mean(-1, keepdim=True) + 1e-6).to(context.dtype)
        scaled = plan.delta_mag_scale * dm
    value = F.silu(xv + dv)
    read = torch.sigmoid(ffn.read_logit + scaled * ffn.read_delta_scale)
    return ffn.output(next_state * value * read), next_state


class EONRMSNorm(nn.Module):
    def __init__(self,dim,eps=1e-6): super().__init__(); self.weight=nn.Parameter(torch.ones(dim)); self.eps=float(eps)
    def forward(self,x): return (x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps).to(x.dtype))*self.weight

class StateAwareFFN(nn.Module):
    def __init__(self,dim,state_dim,depth_dim=64):
        super().__init__(); s=int(state_dim); self.state_dim=s
        self.x_proj=nn.Linear(dim,3*s,bias=True); self.context_candidate=nn.Linear(dim,s,bias=False); self.context_write=nn.Linear(dim,s,bias=False); self.state_proj=nn.Linear(s,2*s,bias=False); self.output=nn.Linear(s,dim,bias=False)
        self.depth_embedding=nn.Parameter(torch.empty(depth_dim)); self.depth_proj=nn.Linear(depth_dim,3*s,bias=False); self.retain_logit=nn.Parameter(torch.full((s,),1.15)); self.read_logit=nn.Parameter(torch.zeros(s))
        self.candidate_transition_logit=nn.Parameter(torch.tensor(-2.0)); self.write_transition_logit=nn.Parameter(torch.tensor(-2.0)); self.retain_delta_scale=nn.Parameter(torch.full((s,),0.10)); self.read_delta_scale=nn.Parameter(torch.full((s,),0.10)); self.delta_magnitude_log_scale=nn.Parameter(torch.tensor(-1.0)); nn.init.normal_(self.depth_embedding,std=0.02)
    def forward(self,x,current_context,previous_context,state):
        s=self.state_dim; xc,xw,xv=self.x_proj(x).split(s,dim=-1); sc,sw=self.state_proj(state).split(s,dim=-1); dc,dw,dv=self.depth_proj(self.depth_embedding).split(s,dim=-1)
        delta=current_context-previous_context; dm=torch.sqrt(delta.float().square().mean(-1,keepdim=True)+1e-6).to(current_context.dtype); scaled=torch.exp(self.delta_magnitude_log_scale)*dm
        cs=torch.sigmoid(self.candidate_transition_logit); ws=torch.sigmoid(self.write_transition_logit); candidate=torch.tanh(xc+self.context_candidate(current_context+cs*delta)+sc+dc); write=torch.sigmoid(xw+self.context_write(current_context+ws*delta)+sw+dw)
        retain=torch.sigmoid(self.retain_logit-scaled*self.retain_delta_scale); next_state=(1.0-write)*(retain*state)+write*candidate; value=F.silu(xv+dv); read=torch.sigmoid(self.read_logit+scaled*self.read_delta_scale)
        return self.output(next_state*value*read),next_state

class EONLayer(nn.Module):
    def __init__(self,dim,state_dim,mixer,ffn): super().__init__(); self.state_dim=int(state_dim); self.norm=EONRMSNorm(dim); self.mixer=mixer; self.ffn=ffn; self.mix=nn.Parameter(torch.tensor(-1.0))
    def forward(self,x,state,previous_context):
        z=self.norm(x); context=self.mixer(z); ffn_out,state=self.ffn(z,context,previous_context,state); return x+torch.sigmoid(self.mix)*(context+ffn_out),state,context

class EONDecodeCache(NamedTuple): mixer_states: tuple[torch.Tensor,...]

def _resolve_mixer(spec,config,dim,backend,precision):
    if isinstance(spec,str):
        cls=_load_mixer_class(spec); cfg=dict(config); cfg.setdefault('embd',dim); cfg.setdefault('dim',dim); cfg.setdefault('d_model',dim); cfg.setdefault('backend',backend); cfg.setdefault('precision',precision)
        if spec.lower()=='esa': cfg.setdefault('head',8); cfg.setdefault('device',None); cfg.setdefault('auto_move_input',False); cfg.setdefault('auto_compile',False)
        return _call_with_supported_kwargs(cls,cfg),spec.lower()
    module=_instantiate_custom(spec,dict(config),kind='mixer') if isinstance(spec,nn.Module) else _instantiate_custom(spec,{**dict(config),'dim':dim},kind='mixer'); return module,module.__class__.__name__

def _resolve_ffn(spec,config,dim,state_dim):
    if isinstance(spec,str):
        key=spec.lower();
        if key=='saffn': return StateAwareFFN(dim,state_dim,**dict(config)),key
        if key=='ffn': return PlainFFNAdapter(ConventionalFFN(dim,**dict(config))),key
        raise ConfigurationError(f"unknown ffn {spec!r}")
    cfg=dict(config); module=_instantiate_custom(spec,cfg,kind='ffn') if isinstance(spec,nn.Module) else _instantiate_custom(spec,{**cfg,'dim':dim,'state_dim':state_dim},kind='ffn'); return (StateAwareFFNAdapter(module) if _is_state_aware(module) else PlainFFNAdapter(module)),module.__class__.__name__

class EON(nn.Module):
    def __init__(self,dim=512,width=32,depth=2,mixer='esa',ffn='saffn',mixer_config=None,ffn_config=None,backend='auto',precision='fp16'):
        super().__init__(); self.dim=int(dim); self.depth=int(depth); self.backend=backend; self.precision=precision
        if self.depth<1: raise ConfigurationError('depth must be >=1')
        self.widths=[int(v) for v in _expand(width,self.depth,'width')]; mixers=_expand(mixer,self.depth,'mixer'); ffns=_expand(ffn,self.depth,'ffn'); mcfgs=_expand_config(mixer_config,self.depth,'mixer_config'); fcfgs=_expand_config(ffn_config,self.depth,'ffn_config')
        layers=[]; self.mixer_names=[]; self.ffn_names=[]
        for i in range(self.depth):
            mx,mn=_resolve_mixer(mixers[i],mcfgs[i],self.dim,backend,precision); fx,fn=_resolve_ffn(ffns[i],fcfgs[i],self.dim,self.widths[i]); layers.append(EONLayer(self.dim,self.widths[i],mx,fx)); self.mixer_names.append(mn); self.ffn_names.append(fn)
        self.layers=nn.ModuleList(layers); self.state_bridges=nn.ModuleList([nn.Identity() if self.widths[i]==self.widths[i+1] else nn.Linear(self.widths[i],self.widths[i+1],bias=False) for i in range(self.depth-1)]); self._uniform_widths=len(set(self.widths))==1
        self._generation_plans=None; self._generation_fast=False
    @property
    def parameter_count(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def _apply(self,fn):
        self.clear_generation_plan(); return super()._apply(fn)
    def load_state_dict(self,*args,**kwargs):
        self.clear_generation_plan(); return super().load_state_dict(*args,**kwargs)
    def set_backend(self,backend):
        self.backend=backend
        for layer in self.layers:
            target=layer.mixer
            if callable(getattr(target,'set_backend',None)): target.set_backend(backend)
            elif hasattr(target,'backend'): target.backend=backend
        return self
    def resolved_backend(self):
        vals=[]
        for layer in self.layers:
            target=layer.mixer
            if callable(getattr(target,'resolved_backend',None)):
                try: vals.append(target.resolved_backend())
                except Exception: pass
            elif hasattr(target,'backend'): vals.append(str(target.backend))
        return vals[0] if vals and all(v==vals[0] for v in vals) else (vals or self.backend)
    def clear_generation_plan(self): self._generation_plans=None; self._generation_fast=False; return self
    def train(self,mode=True):
        if mode: self.clear_generation_plan()
        return super().train(mode)
    def prepare_generation(self,fast=True):
        if self.training: raise RuntimeError('call model.eval() before prepare_generation()')
        self.clear_generation_plan(); self._generation_fast=bool(fast)
        if fast: self._generation_plans=[_build_layer_fast_plan(layer) for layer in self.layers]
        return self
    def _forward_uniform(self,h,state,prev):
        # Equal-width training fast path: skip registered Identity bridges.
        for layer in self.layers:
            h,state,prev=layer(h,state,prev)
        return h,state,prev
    def _forward_bridged(self,h,state,prev):
        # Bridge modules are resolved once at construction.
        for layer,bridge in zip(self.layers[:-1],self.state_bridges):
            h,state,prev=layer(h,state,prev); state=bridge(state)
        h,state,prev=self.layers[-1](h,state,prev)
        return h,state,prev
    def forward(self,x):
        # Keep repeated tensor-contract validation out of the training hot path.
        h=x; state=x.new_zeros(*x.shape[:-1],self.widths[0]); prev=torch.zeros_like(x)
        if self._uniform_widths:
            h,state,prev=self._forward_uniform(h,state,prev)
        else:
            h,state,prev=self._forward_bridged(h,state,prev)
        return h
    @torch.no_grad()
    def prefill(self,x):
        if x.dim()!=3 or x.shape[-1]!=self.dim or x.shape[1]<1: raise ValueError(f"EON prefill input must be [B,T,{self.dim}] with T>=1")
        state=x.new_zeros(*x.shape[:-1],self.widths[0]); prev=torch.zeros_like(x); h=x; states=[]
        for i,layer in enumerate(self.layers):
            if not callable(getattr(layer.mixer,'prefill',None)): raise RuntimeError('mixer must provide prefill() for recurrent generation')
            z=layer.norm(h); context,mstate=layer.mixer.prefill(z); ffn_out,state=layer.ffn(z,context,prev,state); h=h+torch.sigmoid(layer.mix)*(context+ffn_out); prev=context; states.append(mstate)
            if i<len(self.state_bridges): state=self.state_bridges[i](state)
        return h,EONDecodeCache(tuple(states))
    def _decode_standard(self,x,cache):
        h=x; state=x.new_zeros(x.shape[0],1,self.widths[0]); prev=torch.zeros_like(x); states=[]
        for i,layer in enumerate(self.layers):
            z=layer.norm(h); context,mstate=layer.mixer.decode_step(z,cache.mixer_states[i]); ffn_out,state=layer.ffn(z,context,prev,state); h=h+torch.sigmoid(layer.mix)*(context+ffn_out); prev=context; states.append(mstate)
            if i<len(self.state_bridges): state=self.state_bridges[i](state)
        return h,EONDecodeCache(tuple(states))
    def _decode_fast(self,x,cache):
        h=x; prev=None; state=None; states=[]
        for i,plan in enumerate(self._generation_plans):
            z=plan.layer.norm(h); context,mstate,xc,xw,xv=_packed_esa_and_xproj(z,cache.mixer_states[i],plan); ffn_out,state=_fast_saffn(context,prev,state,xc,xw,xv,plan,first_layer=(i==0)); h=h+plan.mix_scale*(context+ffn_out); prev=context; states.append(mstate)
            if i<len(self.state_bridges): state=self.state_bridges[i](state)
        return h,EONDecodeCache(tuple(states))
    def decode_step(self,x,cache):
        if x.dim()==2: x=x.unsqueeze(1)
        if x.dim()!=3 or x.shape[1]!=1 or x.shape[-1]!=self.dim: raise ValueError(f"EON decode_step expects [B,1,{self.dim}]")
        if self._generation_fast and self._generation_plans is not None: return self._decode_fast(x,cache)
        return self._decode_standard(x,cache)
    lightning_prefill=prefill; lightning_step=decode_step
    @torch.no_grad()
    def validate(self,x):
        if not isinstance(x,torch.Tensor): raise TypeError('EON validation input must be a torch.Tensor')
        if x.dim()!=3: raise ValueError(f'EON input must have shape [B,T,D], got {tuple(x.shape)}')
        if x.shape[-1]!=self.dim: raise ValueError(f'EON expected dim={self.dim}, got {x.shape[-1]}')
        y=self.forward(x)
        return {'ok':True,'input_shape':tuple(x.shape),'output_shape':tuple(y.shape),'execution_path':'uniform' if self._uniform_widths else 'bridged'}

def eon(dim=512,width=32,depth=2,mixer='esa',ffn='saffn',mixer_config=None,ffn_config=None,backend='auto',precision='fp16',**kwargs):
    return EON(dim=dim,width=width,depth=depth,mixer=mixer,ffn=ffn,mixer_config=mixer_config,ffn_config=ffn_config,backend=backend,precision=precision,**kwargs)
