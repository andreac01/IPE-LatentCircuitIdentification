from functools import partial
import json

class recursive_forward_discovery:
    def __init__(self, model, prompts, targets, cf_prompts, cf_targets, absolute=True, base_th=0.0025, n_steps=10):
        model.reset_hooks(including_permanent=True)
        self.model = model
        self.prompts = prompts
        self.targets = targets
        self.cf_prompts = cf_prompts
        self.cf_targets = cf_targets
        self.base_th = base_th
        self.n_steps = n_steps
        self.absolute = absolute

        self.clean_logits = model.run_with_hooks(self.model.to_tokens(prompts, prepend_bos=True))
        self.cf_logits, self.cf_cache = model.run_with_cache(self.model.to_tokens(cf_prompts, prepend_bos=True))
        assert self.clean_logits.shape == self.cf_logits.shape, "Clean and CF logits have different shapes"

        self.target_tokens = model.to_tokens(targets, prepend_bos=False)[0, 0]
        self.cf_target_tokens = model.to_tokens(cf_targets, prepend_bos=False)[0, 0]

        self.relevant_nodes = {}
        self.removed_nodes = {}
        self.result_dict = {}
        self.logit_diff = None  # Initialize logit_diff to avoid AttributeError

        self.update_logit_diff(self.clean_logits)
        print("Baseline logit difference: ", self.baseline_logit_diff)


    def logit_diff_base(self, logits, target_tokens, cf_tokens, baseline):
        cur_logit = logits[:, -1, target_tokens].mean().item()
        cur_cf_logit = logits[:, -1, cf_tokens].mean().item()
        cur_diff = cur_logit - cur_cf_logit
        return baseline - cur_diff

    def update_logit_diff(self, logits):
        self.baseline_logit_diff = -self.logit_diff_base(logits, target_tokens=self.target_tokens, cf_tokens=self.cf_target_tokens, baseline=0)
        if not self.absolute or self.logit_diff is None:
            self.logit_diff = partial(self.logit_diff_base, target_tokens=self.target_tokens, cf_tokens=self.cf_target_tokens, baseline=self.baseline_logit_diff)

    def verify_removal(self, hook_name, threshold, head_idx=None):
        self.model.reset_hooks()
        if 'attn' in hook_name:
            assert head_idx is not None, "Head index must be provided for attention hooks"
            hook_fn = partial(self.ablate_head_hook_cf, head_idx=head_idx, cf_cache=self.cf_cache)

        else:
            hook_fn = partial(self.ablation_hook_cf, cf_cache=self.cf_cache)
        self.model.add_hook(hook_name, hook_fn)

        corr_logits = self.model.run_with_hooks(self.prompts, prepend_bos=True)
        logit_diff = self.logit_diff(corr_logits)
        if self.absolute:
            logit_diff = abs(logit_diff)
        if logit_diff < threshold:
            self.model.add_perma_hook(hook_name, hook_fn)
            self.update_logit_diff(corr_logits)
            self.add_node('removed_nodes', hook_name, threshold, head=head_idx)
        else:
            self.add_node('relevant_nodes', hook_name, threshold, head=head_idx)

    def add_node(self, dict_name, hook_name, threshold, head=None):
        node_info = {"hook_name": hook_name, "threshold": threshold}
        if 'attn' in hook_name:
            assert head is not None, "Head index must be provided for attention hooks"
            layer = int(hook_name.split('.')[1])
            if hook_name.split('.')[-1] == 'hook_q':
                mib_name = f'a{layer}.h{head}<q>'
            else:
                if hook_name.split('.')[-1] == 'hook_k':
                    mib_name = f'a{layer}.h{head}<k>'
                else:
                    mib_name = f'a{layer}.h{head}<v>'
        elif 'mlp' in hook_name:
            layer = int(hook_name.split('.')[1])
            mib_name = f'm{layer}'
        elif 'embed' in hook_name:
            mib_name = 'embed'
        node = None
        if dict_name == 'removed_nodes':
            self.removed_nodes[mib_name] = {"node": node, "mib_name": mib_name, "hook_name": hook_name, "node_info": node_info}
        elif dict_name == 'relevant_nodes':
            self.relevant_nodes[mib_name] = {"node": node, "mib_name": mib_name, "hook_name": hook_name, "node_info": node_info}
        else:
            raise ValueError(f"Unknown dict_name: {dict_name}")

    def discover(self):
        thresholds = [self.base_th * 2**i for i in range(self.n_steps)]
        if not self.absolute:
            thresholds = [-th for th in sorted(thresholds, reverse=True)] + sorted(thresholds, reverse=False)
        for threshold in thresholds:
            self.model.reset_hooks()
            self.logits = self.model.run_with_hooks(self.prompts, prepend_bos=True)
            self.update_logit_diff(self.logits)

            if 'embed' not in self.removed_nodes:
                self.verify_removal('hook_embed', threshold)
            for layer in range(self.model.cfg.n_layers-1, -1, -1):
                if f'm{layer}' not in self.removed_nodes:
                    self.verify_removal(f'blocks.{layer}.hook_mlp_out', threshold)
                for head in range(self.model.cfg.n_heads):
                    if f'a{layer}.h{head}<q>' not in self.removed_nodes:
                        self.verify_removal(f'blocks.{layer}.attn.hook_q', threshold, head_idx=head)
                    if f'a{layer}.h{head}<k>' not in self.removed_nodes:
                        self.verify_removal(f'blocks.{layer}.attn.hook_k', threshold, head_idx=head)
                    if f'a{layer}.h{head}<v>' not in self.removed_nodes:
                        self.verify_removal(f'blocks.{layer}.attn.hook_v', threshold, head_idx=head)

            self.result_dict[threshold] = {
                'logit_diff': self.baseline_logit_diff,
                'removed_nodes': self.removed_nodes.copy(),
                'relevant_nodes': self.relevant_nodes.copy(),
                'hooks': self.model.hooks()
            }
            print(f"Threshold {threshold} -> logit diff: {self.baseline_logit_diff}, removed nodes: {len(self.removed_nodes)}, relevant nodes: {len(self.relevant_nodes)}\n")
            self.relevant_nodes = {}
    
    @staticmethod
    def ablation_hook_cf(residual, hook, cf_cache):
        residual[:] = cf_cache[hook.name].detach().clone()
        return residual

    @staticmethod
    def ablate_head_hook_cf(residual, hook, head_idx, cf_cache):
        residual[:, :, head_idx, :] = cf_cache[hook.name][:, :, head_idx, :].detach().clone()
        return residual

    def save_results_mib_json(self, filepath):
        all_relevant_nodes = set()
        for th in self.result_dict.keys():
            nodes = self.result_dict[th]['relevant_nodes'].keys()
            all_relevant_nodes.update(nodes)

        all_relevant_nodes = [k.replace('embed', 'input') for k in all_relevant_nodes] + ['logits']
        all_relevant_nodes_no_qkv = sorted(list(set([node.split('<')[0] for node in all_relevant_nodes])))

        res = {
            "cfg": {
                "n_layers": self.model.cfg.n_layers,
                "n_heads": self.model.cfg.n_heads,
                "d_model": self.model.cfg.d_model,
                "parallel_attn_mlp": self.model.cfg.parallel_attn_mlp
            },
            "nodes": {
                node: {
                    "in_graph": True
                } for node in all_relevant_nodes_no_qkv
            },
            "edges": {}
        }

        def is_after(node1, node2):
            if node1 == "input" or node2 == "logits":
                return True
            elif node1 == "logits" or node2 == "input":
                return False
            
            layer1 = int(node1[1:].split('.')[0])
            layer2 = int(node2[1:].split('.')[0])
            
            if layer1 > layer2:
                return False
            if layer1 == layer2:
                return node1[0] < node2[0]  # 'a' for attention comes before 'm' for mlp
            return True

        for th in self.result_dict.keys():
            relevant_nodes = self.result_dict[th]['relevant_nodes'].keys()
            relevant_nodes = [k.replace('embed', 'input') for k in relevant_nodes]
            relevant_nodes_no_qkv = [node.split('<')[0] for node in relevant_nodes]
            relevant_nodes.append('logits')
            for node1 in relevant_nodes_no_qkv:
                for node2 in relevant_nodes:
                    node2_no_qkv = node2.split('<')[0]
                    if node1 != node2_no_qkv and is_after(node1, node2_no_qkv):
                        res['edges'][f"{node1}->{node2}"] = {
                            "score": th,
                            "in_graph": True
                        }

        with open(filepath, 'w') as f:
            json.dump(res, f, indent=4)