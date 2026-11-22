from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from layers.SRA_layers import SRA_Block
from layers.heads import PretrainHead, ForecastHead, ClfHead
from layers.Embed import TokenEmbeddingFixed
from layers.snippets import split_heads
from sklearn.metrics import average_precision_score


class TrajGPT(nn.Module):
    '''
    TrajGPT leverages Selective Recurrent Attention (SRA) module for irregularly-sampled time series
    '''
    def __init__(self, configs, head_type='pretrain', forecast_method='time_specific', n_output=2,
                 embedding_path=None, embedding_data=None):
        super(TrajGPT, self).__init__()

        # load parameters
        self.num_layers = configs.num_layers
        self.num_heads = configs.num_heads
        self.d_model = configs.d_model
        self.qk_dim = configs.qk_dim
        self.v_dim = configs.v_dim if configs.v_dim else self.qk_dim
        self.dropout = configs.dropout
        self.time_mode = getattr(configs, 'time_mode', 'event_order')
        self.time_scale = float(getattr(configs, 'time_scale', 1.0))
        self.time_unit = getattr(configs, 'time_unit', 'months')
        self.context_length = getattr(configs, 'seq_len', 256)
        if (not isinstance(self.context_length, int) or isinstance(self.context_length, bool)
                or self.context_length < 2):
            raise ValueError('The configured context length must be an integer of at least 2.')
        if self.time_mode not in ('event_order', 'time_conditioned'):
            raise ValueError('time_mode must be event_order or time_conditioned.')
        if self.time_mode == 'time_conditioned':
            if head_type != 'pretrain':
                raise ValueError('Time-conditioned mode is implemented only for the pretraining path.')
            if not math.isfinite(self.time_scale) or self.time_scale <= 0:
                raise ValueError('time_scale must be finite and positive.')
            if self.time_unit != 'months':
                raise ValueError('This PopHR training path expects timestamps in months.')

        # Initialize TokenEmbeddingFixed with pre-trained embeddings
        embedding_path = embedding_path or Path(__file__).resolve().parents[1] / 'data' / 'phecode_embeddings.csv'
        self.token_embedding = TokenEmbeddingFixed(embedding_path, embedding_data=embedding_data)
        if self.token_embedding.emb.embedding_dim != self.d_model:
            raise ValueError('d_model must equal the fixed embedding dimension (200 for the supplied vectors).')
        if self.qk_dim % self.num_heads or self.v_dim % self.num_heads or self.qk_dim % 2:
            raise ValueError('Q/K and V dimensions must be divisible by num_heads; Q/K dimension must be even.')
        if self.time_mode == 'time_conditioned' and (self.qk_dim // self.num_heads) % 2:
            raise ValueError('Time-conditioned RoPE requires an even Q/K dimension per head.')
        if configs.tau <= 0:
            raise ValueError('tau must be positive.')
        # the start token for shifted right
        self.sos = torch.nn.Parameter(torch.zeros(self.d_model))
        nn.init.normal_(self.sos)

        # The stacked SRA blocks, each consists of an SRA module and a MLP layer
        self.SRA_blocks = nn.ModuleList([SRA_Block(configs) for _ in range(self.num_layers)])

        # output layer
        self.ln_f = nn.LayerNorm(self.d_model)  # Layer Normalization
        self.head_type = head_type
        self.forecast_method = forecast_method
        if self.head_type == "pretrain":
            self.n_output = self.token_embedding.get_num_tokens()  # Update n_output to the number of unique PheCodes
            self.head = PretrainHead(self.d_model, self.n_output) # the token is [batch_size x seq_len x c_in]
        elif self.head_type == "forecast":
            self.n_output = self.token_embedding.get_num_tokens()  # Update n_output to the number of unique PheCodes
            self.head = ForecastHead(self.d_model, self.n_output)
        elif self.head_type == "clf":
            self.n_output = n_output # Update the number of output classes
            self.head = ClfHead(self.d_model, self.n_output)
        else:
            raise ValueError("Invalid head_type provided.")
        self.gradient_checkpointing = getattr(configs, 'use_grad_ckp', False)
        if self.time_mode == 'time_conditioned':
            # Query time affects only the local readout, never historical states.
            self.target_time_projection = nn.Linear(1, self.d_model, bias=False)

    @torch.inference_mode()
    def inference(self, history_codes, history_times, future_times,
                  context_length=None, top_k=5):
        """Generate one PheCode at each supplied future event time.

        Inputs are single, unbatched lists/tuples or one-dimensional tensors.
        The caller supplies actual event codes without SOS or PAD. Future times
        are known conditions, not predictions. True future codes are not accepted.
        Each step reuses the pretraining head and appends its own greedy result.

        Returns lists of ``phecodes``, ``times``, ``top_k_phecodes`` and
        ``top_k_probabilities``. Probabilities are normalized over valid diagnosis
        codes only (PAD excluded). Ties follow the saved vocabulary order.
        """
        if self.time_mode != 'time_conditioned' or self.head_type != 'pretrain':
            raise ValueError('inference requires a time-conditioned pretraining model.')
        if context_length is None:
            context_length = self.context_length
        if (not isinstance(context_length, int) or isinstance(context_length, bool)
                or not 2 <= context_length <= self.context_length):
            raise ValueError('context_length must be between 2 and the checkpoint context length.')
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError('top_k must be a positive integer.')

        def copied_sequence(values, name):
            if isinstance(values, torch.Tensor):
                if values.ndim != 1:
                    raise ValueError(f'{name} must be a one-dimensional sequence.')
                return values.detach().cpu().tolist()
            if not isinstance(values, (list, tuple)):
                raise ValueError(f'{name} must be a list, tuple or one-dimensional tensor.')
            return list(values)

        vocabulary = list(self.token_embedding.phecodes)
        valid_codes = set(vocabulary)
        codes = copied_sequence(history_codes, 'history_codes')
        times = copied_sequence(history_times, 'history_times')
        queries = copied_sequence(future_times, 'future_times')
        if not codes or len(codes) != len(times):
            raise ValueError('History codes and times must have the same nonzero length.')
        for index, code in enumerate(codes):
            number = float(code)
            # Membership also rejects fractional, non-finite, SOS and PAD values.
            if isinstance(code, bool) or number not in valid_codes:
                raise ValueError('History must contain known positive integer PheCodes, without SOS or PAD.')
            codes[index] = int(number)
        for values, name in ((times, 'history_times'), (queries, 'future_times')):
            for index, value in enumerate(values):
                number = float(value)
                if isinstance(value, bool) or not math.isfinite(number) or number < 0:
                    raise ValueError(f'{name} must contain finite nonnegative times in months.')
                values[index] = number
            if any(later < earlier for earlier, later in zip(values, values[1:])):
                raise ValueError(f'{name} must be nondecreasing; equal times are allowed.')
        if queries and queries[0] < times[-1]:
            raise ValueError('The first future time must be at or after the last history time.')

        top_k = min(top_k, len(vocabulary))
        result = {'phecodes': [], 'times': [], 'top_k_phecodes': [],
                  'top_k_probabilities': []}
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            for query_time in queries:
                codes = codes[-(context_length - 1):]
                times = times[-(context_length - 1):]
                inputs = torch.tensor([[-1] + codes], dtype=torch.long, device=device)
                # SOS shares the first retained event time, exactly as in training.
                input_time = torch.tensor([[times[0]] + times], dtype=torch.float32, device=device)
                target_time = torch.tensor([times + [query_time]], dtype=torch.float32, device=device)
                output = self(
                    X=inputs, input_time=input_time, target_time=target_time,
                    retention_mask=torch.ones_like(inputs, dtype=torch.bool),
                    forward_impl='parallel',
                )
                # This single unpadded prefix ends at its last valid event.
                # Vocabulary entries precede PAD in TokenEmbeddingFixed.
                scores = output['logits'][0, -1, :len(vocabulary)].float()
                if not torch.isfinite(scores).all().item():
                    raise RuntimeError('Non-finite inference logits for valid PheCodes.')
                ranked = torch.argsort(scores, descending=True, stable=True)[:top_k]
                candidates = [int(vocabulary[index]) for index in ranked.tolist()]
                probabilities = torch.softmax(scores, dim=-1)[ranked].tolist()
                predicted_code = candidates[0]
                result['phecodes'].append(predicted_code)
                result['times'].append(query_time)
                result['top_k_phecodes'].append(candidates)
                result['top_k_probabilities'].append(probabilities)
                codes.append(predicted_code)
                times.append(query_time)
        finally:
            self.train(was_training)
        return result

    def _forward_pretrain(self, X, input_time, target_time, y, retention_mask,
                          forward_impl, past_key_values, sequence_offset,
                          output_retentions, output_hidden_states):
        """Score already-constructed inputs; never shift or drop an event.

        Training constructs [SOS, C, D, E] -> [C, D, E, F]. Generation
        supplies [SOS, C, D, E, F] and reads the last valid position.
        Raw SOS=-1 and PAD=0 are distinct from actual PheCodes.
        """
        if forward_impl != 'parallel' or past_key_values is not None:
            raise ValueError('Basic autoregression uses parallel prefix recomputation; recurrent cache is not supported.')
        if X.ndim != 2 or X.size(1) == 0:
            raise ValueError('X must have shape [batch, nonempty sequence].')
        if not torch.all(X[:, 0].eq(-1)) or torch.any(X[:, 1:].eq(-1)):
            raise ValueError('Inputs must start with exactly one SOS (-1); construct teacher-forcing inputs before calling the model.')
        if input_time is not None and input_time.shape != X.shape:
            raise ValueError('input_time must align with input tokens.')
        if y is not None and y.shape != X.shape:
            raise ValueError('Teacher-forcing targets must align with input tokens.')
        if retention_mask is None:
            retention_mask = X.ne(0)
        elif retention_mask.shape != X.shape:
            raise ValueError('retention_mask must align with input tokens.')
        else:
            retention_mask = retention_mask.to(device=X.device, dtype=torch.bool) & X.ne(0)
        if y is not None and torch.any(y.ne(0) & ~retention_mask):
            raise ValueError('A non-PAD target cannot correspond to a masked input position.')
        if self.gradient_checkpointing and self.training and output_retentions:
            raise ValueError('Attention output is not supported together with gradient checkpointing.')

        time_features = None
        if self.time_mode == 'time_conditioned':
            if input_time is None or target_time is None:
                raise ValueError('Time-conditioned pretraining requires both input_time and target_time; the event-only generation entry point is not compatible.')
            if target_time.shape != X.shape:
                raise ValueError('target_time must align with the prediction targets.')
            # This workflow uses a nonempty prefix followed by right padding.
            if not retention_mask[:, 0].all() or torch.any(retention_mask[:, 1:] & ~retention_mask[:, :-1]):
                raise ValueError('Time-conditioned inputs require valid SOS and right padding only.')
            input_time = input_time.to(device=X.device, dtype=torch.float32)
            target_time = target_time.to(device=X.device, dtype=torch.float32)
            if not torch.isfinite(input_time[retention_mask]).all() or not torch.isfinite(target_time[retention_mask]).all():
                raise ValueError('Valid history and target timestamps must be finite.')
            if torch.any(input_time[retention_mask] < 0) or torch.any(target_time[retention_mask] < 0):
                raise ValueError('PopHR age timestamps must be nonnegative months.')
            adjacent_valid = retention_mask[:, 1:] & retention_mask[:, :-1]
            if torch.any((input_time[:, 1:] < input_time[:, :-1]) & adjacent_valid):
                raise ValueError('Valid historical timestamps must be nondecreasing; equal times are allowed.')
            if torch.any((target_time < input_time) & retention_mask):
                raise ValueError('Each target_time must be at or after its corresponding input_time.')
            # Ignore padded timestamps, including NaN placeholders, before RoPE.
            input_time = torch.where(retention_mask, input_time, input_time[:, :1])
            target_time = torch.where(retention_mask, target_time, input_time)
            # All SRA layers use these same scaled times; validate them once.
            if not torch.isfinite((input_time - input_time[:, :1]) / self.time_scale).all():
                raise ValueError('History time intervals overflowed; check time units and time_scale.')
            delta = (target_time - input_time) / self.time_scale
            time_features = torch.log1p(delta).unsqueeze(-1)
            if not torch.isfinite(time_features).all():
                raise ValueError('Target time intervals overflowed; check time units and time_scale.')

        # Substitute the learned SOS vector without changing sequence length.
        sos_mask = X.eq(-1)
        tokens = X.masked_fill(sos_mask, 0)
        hidden_states = self.token_embedding(tokens)
        hidden_states = torch.where(sos_mask.unsqueeze(-1), self.sos.view(1, 1, -1), hidden_states)
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        for block in self.SRA_blocks:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                # Bind the current layer; backward may run after this loop ends.
                def custom_forward(states, current_block=block):
                    return current_block(
                        states, input_time, retention_mask=retention_mask,
                        forward_impl='parallel', sequence_offset=sequence_offset,
                        output_retentions=False, use_cache=False)[0]
                hidden_states = torch.utils.checkpoint.checkpoint(
                    custom_forward, hidden_states, use_reentrant=False)
            else:
                block_outputs = block(
                    hidden_states, input_time, retention_mask=retention_mask,
                    forward_impl='parallel', sequence_offset=sequence_offset,
                    output_retentions=output_retentions, use_cache=False)
                hidden_states = block_outputs[0]
                if output_retentions:
                    all_retentions += (block_outputs[2],)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        if time_features is not None:
            hidden_states = hidden_states + self.target_time_projection(
                time_features.to(dtype=hidden_states.dtype))
        logits = self.head(self.ln_f(hidden_states))
        result = {'logits': logits,
                  'loss': self.compute_pretrain_loss(logits, y) if y is not None else None}
        if output_hidden_states:
            result['hidden_states'] = all_hidden_states
        if output_retentions:
            result['retentions'] = all_retentions
        return result

    def forward(self,
                X, input_time=None,
                y=None, target_time=None,
                retention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                forward_impl: Optional[str] = 'parallel',
                sequence_offset: Optional[int] = 0,
                output_retentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                ):
        if self.head_type == 'pretrain':
            return self._forward_pretrain(
                X, input_time, target_time, y, retention_mask, forward_impl, past_key_values,
                sequence_offset, output_retentions, output_hidden_states)

        # Legacy downstream heads are retained; the basic generator deliberately
        # reuses the trained pretrain head instead of this forecasting branch.
        batch_size, seq_len = X.shape
        # transform a sequence of tokens to embeddings
        hidden_states = torch.zeros(batch_size, seq_len, self.d_model).to(X.device) # batch_size x seq_len x d_model
        for i in range(batch_size):
            # Use token_embedding to get embeddings for the entire sequence for batch i
            hidden_states[i] = self.token_embedding(X[i])
        # Add the SOS token to the input sequence
        sos_token = self.sos.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape [batch_size, 1, d_model]
        hidden_states = torch.cat([sos_token, hidden_states[:, :-1, :]], dim=1)  # Shift right and drop the last value to maintain original length

        if retention_mask is None:
            retention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=X.device) # batch_size x token_num
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        present_key_values = ()  # To store current key-value pairs
        for i, block in enumerate(self.SRA_blocks):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training: # Use gradient checkpointing for the forward pass
                def custom_forward(*inputs):
                    return block(*inputs)
                block_outputs = torch.utils.checkpoint.checkpoint(
                                custom_forward,
                                hidden_states,
                                input_time,
                                retention_mask,
                                forward_impl,
                                past_key_value,
                                sequence_offset,
                                output_retentions
                                )
            else:
                block_outputs = block(hidden_states,
                                      input_time,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      sequence_offset=sequence_offset,
                                      output_retentions=output_retentions
                                      )
            # outputs two variables if output_retentions is False: output hidden states (self.proj(out)), present key values (curr_kv)
            hidden_states = block_outputs[0]
            present_key_values += (block_outputs[1],)

            # if output_retentions is True, we have an extra variable in block_outputs: retentions (visualization analysis)
            if output_retentions:
                all_retentions += (block_outputs[2],)

        # add hidden states from the last layer
        if output_hidden_states:
            all_hidden_states += (hidden_states)

        # Apply the custom head on the hidden states for output
        outputs = self.ln_f(hidden_states)
        if self.head_type == "forecast":
            logits_seq = self._run_forecast(hidden_states, input_time, target_time, y)
            logits = torch.stack(logits_seq, dim=1) # stack into (B, M, vocab_size) and compute loss
            return self.compute_forecast_loss(logits, y)
        elif self.head_type == 'clf':
            logits = self.head(outputs)
            return self.compute_classify_loss(logits, y) # return classification loss
        elif self.head_type == 'multi_target_clf': # return AUPRC
            logits = self.head(outputs)
            return self.compute_multi_target_clf_loss(logits, y) # return multi-target classification loss

    def _run_forecast(self,
                      hidden_states: torch.Tensor,
                      input_time:    torch.Tensor,
                      target_time:   torch.Tensor
                     ) -> List[torch.Tensor]:
        """
        Performs M-step forecasting using SRA.recurrent_retention.
        Returns a list of `M` per-step logits of shape (B, n_output).
        """
        B, T, D     = hidden_states.shape
        M = target_time.size(1) if target_time is not None else 1
        qkv = self.SRA_blocks[-1].sra.qkv(hidden_states)

        q, k, v = torch.split(qkv,
                              [self.qk_dim, self.qk_dim, self.v_dim],
                              dim=-1)
        # split into heads: each becomes (B, H, T, *)
        q, k, v = split_heads((q, k, v), B, T, self.num_heads)
        # only keep the last time step
        qN, kN, vN = q[:, :, -1:], k[:, :, -1:], v[:, :, -1:]
        state = hidden_states[:, -1, :]

        if self.forecast_method == "time_specific":
            # Obtain the data‐dependent decay at time step N from SRA
            gamma = self.SRA_blocks[-1].sra.get_data_dependent_decay(hidden_states)  # (B, T, H)
            gamma_N = gamma[:, -1, :].view(B, self.num_heads, 1, 1)
            last_t = input_time[:, -1]  # shape (B,)
            dt = target_time - last_t.unsqueeze(1)  # shape (B, M)
        else:
            gamma_N = None
            dt = torch.ones(B, M, device=hidden_states.device)

        # loop M steps of recurrence and projection
        logits_seq = []
        for m in range(M):
            if self.forecast_method == "time_specific":
                decay_m = gamma_N ** dt[:, m].view(B, self.num_heads, 1, 1) # For time-specific inference, it will the decay rate at the time N
            else:
                decay_m = None  # For auto-regressive inference, it will the previous decay rate
            # recurrent update: (qN,kN,vN) are each (B,H,1,*)
            out, S = self.SRA_blocks[-1].sra.recurrent_retention(
                qN, kN, vN,
                past_key_value=state,
                decay=decay_m,
                retention_mask=None
            )
            hidden_out = out.permute(0, 2, 1, 3).reshape(B, -1)
            logits_seq.append(self.head(hidden_out))

        return logits_seq


    def compute_pretrain_loss(self, logits, targets):
        """
        Compute the loss of the pre-training task (next token prediction)
        """
        targets = self.token_embedding.map_phecodes_to_indices(targets.reshape(-1))
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets,
                                   ignore_index=self.token_embedding.padding_idx,
                                   reduction='sum')
        valid_count = targets.ne(self.token_embedding.padding_idx).sum()
        return loss_sum / valid_count.clamp_min(1)

    def compute_forecast_loss(self, logits, targets):
        """
        Compute the cross entropy loss for the forecasting head
        """
        # Define CrossEntropyLoss with ignore_index for padding token
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.token_embedding.padding_idx)
        # logits: [batch_size, seq_len, num_classes], targets: [batch_size, seq_len]
        logits = logits.view(-1, logits.size(-1))
        # Convert PheCode targets to indices
        targets = self.token_embedding.map_phecodes_to_indices(targets.view(-1)).to(logits.device)
        token_loss = self.ce_loss(logits, targets)
        return token_loss

    def compute_classify_loss(self, logits, targets):
        """
        Compute the loss of classification task
        """
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)
        # Apply a threshold of 0.5 to get binary predictions
        predicted = (probs > 0.5).float()
        # Compute accuracy
        correct = (predicted == targets).float()
        accuracy = correct.mean() * 100.0
        return accuracy

    def compute_multi_target_clf_loss(self, logits, targets):
        """
        Compute the loss of multi-target classification task
        """
        # compute binary cross entropy with logits loss
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss()
        loss = self.bce_with_logits_loss(logits, targets)

        # Threshold logits to get predicted labels
        predicted = (logits > 0.5).float()

        # compute Precision, Recall
        TP = (predicted * targets).sum().float()
        FP = (predicted * (1 - targets)).sum().float()
        FN = ((1 - predicted) * targets).sum().float()

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)

        # AUPRC
        y_true = targets.cpu().numpy()
        y_scores = logits.cpu().numpy()
        auprc = average_precision_score(y_true, y_scores)

        return loss, precision, recall, auprc
