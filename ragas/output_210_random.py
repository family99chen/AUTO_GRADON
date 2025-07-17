[06/15/25 18:51:22] INFO     [config.py:58] >> PyTorch version               config.py:58
                             2.8.0.dev20250521+cu128 available.                          
[06/15/25 18:51:25] INFO     [_client.py:1026] >> HTTP Request: GET       _client.py:1026
                             https://api.gradio.app/gradio-messaging/en                  
                             "HTTP/1.1 200 OK"                                           
[06/15/25 18:51:39] INFO     [random_selection_evaluate. random_selection_evaluate.py:123
                             py:123] >> 日志文件保存在:                                  
                             ../experiments/210-random_c                                 
                             omparison/random_strategy_o                                 
                             pt/logs/random_optimization                                 
                             _20250615_185139.log                                        
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:152
                             py:152] >> 将优化以下 5                                     
                             个组件: ['retrieval',                                       
                             'query_expansion',                                          
                             'passage_reranker',                                         
                             'passage_filter',                                           
                             'passage_compressor']                                       
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:180
                             py:180] >>                                                  
                             初始化了5个节点的配置:                                      
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:182
                             py:182] >>   node1: {'0':                                   
                             'hybrid_rrf', '1': 'bm25',                                  
                             '2': 'vectordb', '3':                                       
                             'hybrid_cc'}                                                
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:182
                             py:182] >>   node2: {'0':                                   
                             'pass_query_expansion',                                     
                             '1': 'QueryDecompose', '2':                                 
                             'HyDE', '3':                                                
                             'multi_query_expansion'}                                    
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:182
                             py:182] >>   node3: {'0':                                   
                             'pass_reranker', '1':                                       
                             'upr', '2': 'tart', '3':                                    
                             'colbert_reranker'}                                         
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:182
                             py:182] >>   node4: {'0':                                   
                             'pass_passage_filter', '1':                                 
                             'similarity_threshold_cutof                                 
                             f', '2':                                                    
                             'similarity_percentile_cuto                                 
                             ff', '3':                                                   
                             'threshold_cutoff'}                                         
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:182
                             py:182] >>   node5: {'0':                                   
                             'pass_compressor', '1':                                     
                             'tree_summarize', '2':                                      
                             'refine', '3':                                              
                             'longllmlingua'}                                            
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:188
                             py:188] >> 总配置空间大小:                                  
                             1024                                                        
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:454
                             py:454] >>                                                  
                             开始随机优化，5个组件，25次                                 
                             试验                                                        
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:464
                             py:464] >>                                                  
                             ==================== Trial                                  
                             1/25 ====================                                   
                    INFO     [random_selection_evaluate.p random_selection_evaluate.py:40
                             y:40] >> Trial 1                                            
                             生成随机配置 耗时: 0.00 秒                                  
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:486
                             py:486] >> Trial 1 配置:                                    
                             {'retrieval': 'vectordb',                                   
                             'query_expansion':                                          
                             'QueryDecompose',                                           
                             'passage_reranker':                                         
                             'pass_reranker',                                            
                             'passage_filter':                                           
                             'pass_passage_filter',                                      
                             'passage_compressor':                                       
                             'tree_summarize'}                                           
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:274
                             py:274] >> 选择的方法:                                      
                             {'retrieval': 'vectordb',                                   
                             'query_expansion':                                          
                             'QueryDecompose',                                           
                             'passage_reranker':                                         
                             'pass_reranker',                                            
                             'passage_filter':                                           
                             'pass_passage_filter',                                      
                             'passage_compressor':                                       
                             'tree_summarize'}                                           
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:335
                             py:335] >> 为组件                                           
                             query_expansion                                             
                             找到匹配方法 QueryDecompose                                 
                             的配置                                                      
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:335
                             py:335] >> 为组件 retrieval                                 
                             找到匹配方法 vectordb                                       
                             的配置                                                      
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:335
                             py:335] >> 为组件                                           
                             passage_reranker                                            
                             找到匹配方法 pass_reranker                                  
                             的配置                                                      
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:335
                             py:335] >> 为组件                                           
                             passage_filter 找到匹配方法                                 
                             pass_passage_filter 的配置                                  
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:335
                             py:335] >> 为组件                                           
                             passage_compressor                                          
                             找到匹配方法 tree_summarize                                 
                             的配置                                                      
                    INFO     [random_selection_evaluate. random_selection_evaluate.py:378
                             py:378] >>                                                  
                             生成配置文件，使用了 8                                      
                             个组件配置                                                  
BM25 corpus embedding complete.
load_vectordb: chroma
[06/15/25 18:51:59] INFO     [SentenceTransformer.py:218] >>   SentenceTransformer.py:218
                             Load pretrained                                             
                             SentenceTransformer:                                        
                             sentence-transformers/all-mpnet-b                           
                             ase-v2                                                      
[06/15/25 18:52:01] INFO     [SentenceTransformer.py:357] >> 2 SentenceTransformer.py:357
                             prompts are loaded, with the                                
                             keys: ['query', 'text']                                     
                    INFO     [posthog.py:22] >> Anonymized telemetry        posthog.py:22
                             enabled. See                                                
                             https://docs.trychroma.com/telemetry for more               
                             information.                                                
[06/15/25 18:52:21] ERROR    [__init__.py:60] >> Unexpected exception      __init__.py:60
                             ╭──── Traceback (most recent call last) ────╮               
                             │ /home/cz/AUTO_GRADON/ragas/random_selecti │               
                             │ on_evaluate.py:621 in <module>            │               
                             │                                           │               
                             │   618 │   )                               │               
                             │   619 │                                   │               
                             │   620 │   # 运行优化，25次不重复的随机试  │               
                             │ ❱ 621 │   best_config, best_reward = opti │               
                             │   622 │                                   │               
                             │   623 │   print(f"\n🎯 随机优化最终结果:" │               
                             │   624 │   print(f"最佳配置: {best_config} │               
                             │                                           │               
                             │ /home/cz/AUTO_GRADON/ragas/random_selecti │               
                             │ on_evaluate.py:490 in run_optimization    │               
                             │                                           │               
                             │   487 │   │   │                           │               
                             │   488 │   │   │   # 评估配置              │               
                             │   489 │   │   │   with self.timer(f"Trial │               
                             │ ❱ 490 │   │   │   │   reward = self.evalu │               
                             │   491 │   │   │                           │               
                             │   492 │   │   │   # 记录结果              │               
                             │   493 │   │   │   trial_time = time.time( │               
                             │                                           │               
                             │ /home/cz/AUTO_GRADON/ragas/random_selecti │               
                             │ on_evaluate.py:417 in evaluate_config     │               
                             │                                           │               
                             │   414 │   │   │   config_file = self.gene │               
                             │   415 │   │   │                           │               
                             │   416 │   │   │   # 初始化runner          │               
                             │ ❱ 417 │   │   │   self.evaluator.init_run │               
                             │   418 │   │   │                           │               
                             │   419 │   │   │   # 评估配置              │               
                             │   420 │   │   │   yaml_name = f"config_{' │               
                             │                                           │               
                             │ /home/cz/AUTO_GRADON/ragas/evaluator.py:1 │               
                             │ 56 in init_runner_from_yaml               │               
                             │                                           │               
                             │   153 │   │   # Ingest VectorDB corpus    │               
                             │   154 │   │   if yaml_dict.get("vectordb" │               
                             │   155 │   │   │   loop = get_event_loop() │               
                             │ ❱ 156 │   │   │   loop.run_until_complete │               
                             │   157 │   │                               │               
                             │   158 │   │   self.runner = Runner.from_y │               
                             │   159 │   │   self.strategy = yaml_dict.g │               
                             │       'bert_score']})                     │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/asyncio/base_events.py:636 in           │               
                             │ run_until_complete                        │               
                             │                                           │               
                             │    633 │   │                              │               
                             │    634 │   │   future.add_done_callback(_ │               
                             │    635 │   │   try:                       │               
                             │ ❱  636 │   │   │   self.run_forever()     │               
                             │    637 │   │   except:                    │               
                             │    638 │   │   │   if new_task and future │               
                             │    639 │   │   │   │   # The coroutine ra │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/asyncio/base_events.py:603 in           │               
                             │ run_forever                               │               
                             │                                           │               
                             │    600 │   │   │                          │               
                             │    601 │   │   │   events._set_running_lo │               
                             │    602 │   │   │   while True:            │               
                             │ ❱  603 │   │   │   │   self._run_once()   │               
                             │    604 │   │   │   │   if self._stopping: │               
                             │    605 │   │   │   │   │   break          │               
                             │    606 │   │   finally:                   │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/asyncio/base_events.py:1909 in          │               
                             │ _run_once                                 │               
                             │                                           │               
                             │   1906 │   │   │   │   finally:           │               
                             │   1907 │   │   │   │   │   self._current_ │               
                             │   1908 │   │   │   else:                  │               
                             │ ❱ 1909 │   │   │   │   handle._run()      │               
                             │   1910 │   │   handle = None  # Needed to │               
                             │   1911 │                                  │               
                             │   1912 │   def _set_coroutine_origin_trac │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/asyncio/events.py:80 in _run            │               
                             │                                           │               
                             │    77 │                                   │               
                             │    78 │   def _run(self):                 │               
                             │    79 │   │   try:                        │               
                             │ ❱  80 │   │   │   self._context.run(self. │               
                             │    81 │   │   except (SystemExit, Keyboar │               
                             │    82 │   │   │   raise                   │               
                             │    83 │   │   except BaseException as exc │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/llama_index/embeddings/hu │               
                             │ ggingface/base.py:255 in                  │               
                             │ _aget_text_embedding                      │               
                             │                                           │               
                             │   252 │   │   Returns:                    │               
                             │   253 │   │   │   List[float]: numpy arra │               
                             │   254 │   │   """                         │               
                             │ ❱ 255 │   │   return self._get_text_embed │               
                             │   256 │                                   │               
                             │   257 │   def _get_text_embedding(self, t │               
                             │   258 │   │   """Generates Embeddings for │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/llama_index/core/instrume │               
                             │ ntation/dispatcher.py:311 in wrapper      │               
                             │                                           │               
                             │   308 │   │   │   │   │   │   _logger.deb │               
                             │   309 │   │   │                           │               
                             │   310 │   │   │   try:                    │               
                             │ ❱ 311 │   │   │   │   result = func(*args │               
                             │   312 │   │   │   │   if isinstance(resul │               
                             │   313 │   │   │   │   │   # If the result │               
                             │   314 │   │   │   │   │   new_future = as │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/llama_index/embeddings/hu │               
                             │ ggingface/base.py:266 in                  │               
                             │ _get_text_embedding                       │               
                             │                                           │               
                             │   263 │   │   Returns:                    │               
                             │   264 │   │   │   List[float]: numpy arra │               
                             │   265 │   │   """                         │               
                             │ ❱ 266 │   │   return self._embed(text, pr │               
                             │   267 │                                   │               
                             │   268 │   def _get_text_embeddings(self,  │               
                             │   269 │   │   """Generates Embeddings for │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/llama_index/embeddings/hu │               
                             │ ggingface/base.py:215 in _embed           │               
                             │                                           │               
                             │   212 │   │   │   self._model.stop_multi_ │               
                             │   213 │   │                               │               
                             │   214 │   │   else:                       │               
                             │ ❱ 215 │   │   │   emb = self._model.encod │               
                             │   216 │   │   │   │   sentences,          │               
                             │   217 │   │   │   │   batch_size=self.emb │               
                             │   218 │   │   │   │   prompt_name=prompt_ │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/sentence_transformers/Sen │               
                             │ tenceTransformer.py:623 in encode         │               
                             │                                           │               
                             │    620 │   │   │   features.update(extra_ │               
                             │    621 │   │   │                          │               
                             │    622 │   │   │   with torch.no_grad():  │               
                             │ ❱  623 │   │   │   │   out_features = sel │               
                             │    624 │   │   │   │   if self.device.typ │               
                             │    625 │   │   │   │   │   out_features = │               
                             │    626                                    │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/sentence_transformers/Sen │               
                             │ tenceTransformer.py:690 in forward        │               
                             │                                           │               
                             │    687 │   │   for module_name, module in │               
                             │    688 │   │   │   module_kwarg_keys = se │               
                             │    689 │   │   │   module_kwargs = {key:  │               
                             │        module_kwarg_keys}                 │               
                             │ ❱  690 │   │   │   input = module(input,  │               
                             │    691 │   │   return input               │               
                             │    692 │                                  │               
                             │    693 │   @property                      │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1767 in _wrapped_call_impl              │               
                             │                                           │               
                             │   1764 │   │   if self._compiled_call_imp │               
                             │   1765 │   │   │   return self._compiled_ │               
                             │   1766 │   │   else:                      │               
                             │ ❱ 1767 │   │   │   return self._call_impl │               
                             │   1768 │                                  │               
                             │   1769 │   # torchrec tests the code cons │               
                             │   1770 │   # fmt: off                     │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1778 in _call_impl                      │               
                             │                                           │               
                             │   1775 │   │   if not (self._backward_hoo │               
                             │        or self._forward_pre_hooks         │               
                             │   1776 │   │   │   │   or _global_backwar │               
                             │   1777 │   │   │   │   or _global_forward │               
                             │ ❱ 1778 │   │   │   return forward_call(*a │               
                             │   1779 │   │                              │               
                             │   1780 │   │   result = None              │               
                             │   1781 │   │   called_always_called_hooks │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/sentence_transformers/mod │               
                             │ els/Transformer.py:393 in forward         │               
                             │                                           │               
                             │   390 │   │   if "token_type_ids" in feat │               
                             │   391 │   │   │   trans_features["token_t │               
                             │   392 │   │                               │               
                             │ ❱ 393 │   │   output_states = self.auto_m │               
                             │   394 │   │   output_tokens = output_stat │               
                             │   395 │   │                               │               
                             │   396 │   │   # If the AutoModel is wrapp │               
                             │       have added virtual tokens           │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1767 in _wrapped_call_impl              │               
                             │                                           │               
                             │   1764 │   │   if self._compiled_call_imp │               
                             │   1765 │   │   │   return self._compiled_ │               
                             │   1766 │   │   else:                      │               
                             │ ❱ 1767 │   │   │   return self._call_impl │               
                             │   1768 │                                  │               
                             │   1769 │   # torchrec tests the code cons │               
                             │   1770 │   # fmt: off                     │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1778 in _call_impl                      │               
                             │                                           │               
                             │   1775 │   │   if not (self._backward_hoo │               
                             │        or self._forward_pre_hooks         │               
                             │   1776 │   │   │   │   or _global_backwar │               
                             │   1777 │   │   │   │   or _global_forward │               
                             │ ❱ 1778 │   │   │   return forward_call(*a │               
                             │   1779 │   │                              │               
                             │   1780 │   │   result = None              │               
                             │   1781 │   │   called_always_called_hooks │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/transformers/models/mpnet │               
                             │ /modeling_mpnet.py:544 in forward         │               
                             │                                           │               
                             │    541 │   │                              │               
                             │    542 │   │   head_mask = self.get_head_ │               
                             │    543 │   │   embedding_output = self.em │               
                             │        position_ids=position_ids, inputs_ │               
                             │ ❱  544 │   │   encoder_outputs = self.enc │               
                             │    545 │   │   │   embedding_output,      │               
                             │    546 │   │   │   attention_mask=extende │               
                             │    547 │   │   │   head_mask=head_mask,   │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1767 in _wrapped_call_impl              │               
                             │                                           │               
                             │   1764 │   │   if self._compiled_call_imp │               
                             │   1765 │   │   │   return self._compiled_ │               
                             │   1766 │   │   else:                      │               
                             │ ❱ 1767 │   │   │   return self._call_impl │               
                             │   1768 │                                  │               
                             │   1769 │   # torchrec tests the code cons │               
                             │   1770 │   # fmt: off                     │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1778 in _call_impl                      │               
                             │                                           │               
                             │   1775 │   │   if not (self._backward_hoo │               
                             │        or self._forward_pre_hooks         │               
                             │   1776 │   │   │   │   or _global_backwar │               
                             │   1777 │   │   │   │   or _global_forward │               
                             │ ❱ 1778 │   │   │   return forward_call(*a │               
                             │   1779 │   │                              │               
                             │   1780 │   │   result = None              │               
                             │   1781 │   │   called_always_called_hooks │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/transformers/models/mpnet │               
                             │ /modeling_mpnet.py:334 in forward         │               
                             │                                           │               
                             │    331 │   │   │   if output_hidden_state │               
                             │    332 │   │   │   │   all_hidden_states  │               
                             │    333 │   │   │                          │               
                             │ ❱  334 │   │   │   layer_outputs = layer_ │               
                             │    335 │   │   │   │   hidden_states,     │               
                             │    336 │   │   │   │   attention_mask,    │               
                             │    337 │   │   │   │   head_mask[i],      │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1767 in _wrapped_call_impl              │               
                             │                                           │               
                             │   1764 │   │   if self._compiled_call_imp │               
                             │   1765 │   │   │   return self._compiled_ │               
                             │   1766 │   │   else:                      │               
                             │ ❱ 1767 │   │   │   return self._call_impl │               
                             │   1768 │                                  │               
                             │   1769 │   # torchrec tests the code cons │               
                             │   1770 │   # fmt: off                     │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1778 in _call_impl                      │               
                             │                                           │               
                             │   1775 │   │   if not (self._backward_hoo │               
                             │        or self._forward_pre_hooks         │               
                             │   1776 │   │   │   │   or _global_backwar │               
                             │   1777 │   │   │   │   or _global_forward │               
                             │ ❱ 1778 │   │   │   return forward_call(*a │               
                             │   1779 │   │                              │               
                             │   1780 │   │   result = None              │               
                             │   1781 │   │   called_always_called_hooks │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/transformers/models/mpnet │               
                             │ /modeling_mpnet.py:293 in forward         │               
                             │                                           │               
                             │    290 │   │   output_attentions=False,   │               
                             │    291 │   │   **kwargs,                  │               
                             │    292 │   ):                             │               
                             │ ❱  293 │   │   self_attention_outputs = s │               
                             │    294 │   │   │   hidden_states,         │               
                             │    295 │   │   │   attention_mask,        │               
                             │    296 │   │   │   head_mask,             │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1767 in _wrapped_call_impl              │               
                             │                                           │               
                             │   1764 │   │   if self._compiled_call_imp │               
                             │   1765 │   │   │   return self._compiled_ │               
                             │   1766 │   │   else:                      │               
                             │ ❱ 1767 │   │   │   return self._call_impl │               
                             │   1768 │                                  │               
                             │   1769 │   # torchrec tests the code cons │               
                             │   1770 │   # fmt: off                     │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1778 in _call_impl                      │               
                             │                                           │               
                             │   1775 │   │   if not (self._backward_hoo │               
                             │        or self._forward_pre_hooks         │               
                             │   1776 │   │   │   │   or _global_backwar │               
                             │   1777 │   │   │   │   or _global_forward │               
                             │ ❱ 1778 │   │   │   return forward_call(*a │               
                             │   1779 │   │                              │               
                             │   1780 │   │   result = None              │               
                             │   1781 │   │   called_always_called_hooks │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/transformers/models/mpnet │               
                             │ /modeling_mpnet.py:234 in forward         │               
                             │                                           │               
                             │    231 │   │   output_attentions=False,   │               
                             │    232 │   │   **kwargs,                  │               
                             │    233 │   ):                             │               
                             │ ❱  234 │   │   self_outputs = self.attn(  │               
                             │    235 │   │   │   hidden_states,         │               
                             │    236 │   │   │   attention_mask,        │               
                             │    237 │   │   │   head_mask,             │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1767 in _wrapped_call_impl              │               
                             │                                           │               
                             │   1764 │   │   if self._compiled_call_imp │               
                             │   1765 │   │   │   return self._compiled_ │               
                             │   1766 │   │   else:                      │               
                             │ ❱ 1767 │   │   │   return self._call_impl │               
                             │   1768 │                                  │               
                             │   1769 │   # torchrec tests the code cons │               
                             │   1770 │   # fmt: off                     │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/torch/nn/modules/module.p │               
                             │ y:1778 in _call_impl                      │               
                             │                                           │               
                             │   1775 │   │   if not (self._backward_hoo │               
                             │        or self._forward_pre_hooks         │               
                             │   1776 │   │   │   │   or _global_backwar │               
                             │   1777 │   │   │   │   or _global_forward │               
                             │ ❱ 1778 │   │   │   return forward_call(*a │               
                             │   1779 │   │                              │               
                             │   1780 │   │   result = None              │               
                             │   1781 │   │   called_always_called_hooks │               
                             │                                           │               
                             │ /home/cz/.conda/envs/autodl/lib/python3.1 │               
                             │ 0/site-packages/transformers/models/mpnet │               
                             │ /modeling_mpnet.py:170 in forward         │               
                             │                                           │               
                             │    167 │   │   v = self.transpose_for_sco │               
                             │    168 │   │                              │               
                             │    169 │   │   # Take the dot product bet │               
                             │        scores.                            │               
                             │ ❱  170 │   │   attention_scores = torch.m │               
                             │    171 │   │   attention_scores = attenti │               
                             │    172 │   │                              │               
                             │    173 │   │   # Apply relative position  │               
                             ╰───────────────────────────────────────────╯               
                             KeyboardInterrupt                                           
