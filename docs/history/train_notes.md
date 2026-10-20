
 记忆已保存 ✅

  我把当前进度和全部关键发现写入了 opt/PROGRESS.md（工作区根目录下），下次你说【继续多模态】即可从断点续接。                                                   
                                                                                                                                                               
  本次会话的核心发现（摘要）：                                                                                                                                 
                                                                                                                                                               
  1. 诊断结论：val 集本质是"文本匹配"任务（label=1 ⟺ enrollment 文本包含在 query 文本中，100% 一致），所以本地 CV 高达 0.987 但线上只有 0.82755 —— val 不能作为
     线上难度的可靠代理。                                                                                                                                      
                                                                                                                                                               
  2. Whisper-tiny ASR 完全不可用（幻觉/重复输出，fuzzy-match AUC≈0.5），已放弃 ASR 文本匹配路线。                                                              
  3. 最大机会：500K 训练集从未被用于训练匹配模型（train_feats.npz 不存在），现有 p1–p4 都只在 10K val 上训练，泛化差是必然。                                   
  4. 音频余弦基线：dev_seen AUC≈0.81（难）、dev_unseen≈0.95（易）——需要靠 500K 训练 + 帧级匹配 + 文本条件来提升。                                              
                                                                                                                                                               
  中断点：500K 训练特征提取脚本已写好（opt/extract_train_feats_v2.py，分块+断点续传），但尚未成功启动（启动参数被引号搞坏，需用 Start-Process -ArgumentList    
  @('-u','opt/extract_train_feats_v2.py') 重启）。帧级匹配模型脚本 opt/train_seqmatcher.py 也已就绪待验证。                                                    
                                                                                                                                                               
  另外注意到 GPU 正被另一个 YOLO 训练任务（torch1 环境，PID 40808）占用，恢复时需考虑算力竞争。                                                                
                                                                                                                                                               
  下次说【继续多模态】我会：① 启动特征提取 → ② 在 10K val 验证帧级模型 → ③ 在 500K 上训练并生成新提交。   